"""
DeepStream Best-Frame Selection Pipeline
=========================================
Input  : MP4 file (H.264)
Output : MP4 file (H.264) with OSD overlays
         + out/frames/  — top-3 full frames per tracked object
         + out/crops/   — top-2 bbox crops per tracked object

Termination detection uses the dual-channel approach:
  Channel 1 : NvDsObjectMeta       (active, confirmed detections)
  Channel 2 : NvDsPastFrameObjBatch (shadow-phase, held-back detections)
  Terminated = was in known_ids AND absent from both channels this batch.

Requirements
------------
  DeepStream 8, Python bindings (pyds), GStreamer, OpenCV, NumPy
  PGIE config  : config_pgie.txt   (vehicle detector, class 0 = vehicle)
  Tracker config: config_tracker.yml
      NvDCF:
        outputHistoryData: 1
        maxHistorySize: 5
        maxShadowTrackingAge: 30

Usage
-----
  python deepstream_pipeline.py --input input.mp4 --output output.mp4
"""

import sys
import os
import argparse
import math
from collections import defaultdict

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds
import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

PGIE_CONFIG    = "config_pgie.txt"
TRACKER_CONFIG = "config_tracker.yml"

BUFFER_CAP    = 50    # max frames stored per (src_id, object_id)
TOP_FRAMES    = 3     # full frames saved per track
TOP_CROPS     = 2     # bbox crops saved per track
TIMEOUT_FRAMES = 60   # safety-net flush: frames since last_seen

OUTPUT_FRAMES_DIR = "out/frames"
OUTPUT_CROPS_DIR  = "out/crops"

# ──────────────────────────────────────────────────────────────────────────────
# STATE  (shared across probe calls)
# ──────────────────────────────────────────────────────────────────────────────

object_frame_buffer = defaultdict(list)   # (src_id, oid) → [entry, ...]
last_seen           = {}                  # (src_id, oid) → frame_num
known_ids           = set()              # IDs alive in previous batch


# ──────────────────────────────────────────────────────────────────────────────
# QUALITY SCORING
# ──────────────────────────────────────────────────────────────────────────────

def frame_quality_score(conf, rect, frame_meta):
    """
    Composite score combining:
      0.45 · detector confidence
      0.35 · normalised bbox area  (larger = more detail)
      0.10 · aspect-ratio sanity   (avoids heavily occluded/clipped objects)
      0.10 · centre-margin penalty (avoids edge-truncated objects)
    """
    w = rect.width
    h = rect.height

    aspect_ok  = float(0.4 < (w / max(h, 1)) < 2.5)
    conf_score = conf
    frame_area = (frame_meta.source_frame_width *
                  frame_meta.source_frame_height)
    size_score = min((w * h) / max(frame_area, 1), 1.0)

    cx = rect.left + w / 2
    cy = rect.top  + h / 2
    edge_ok = float(
        0.05 < cx / max(frame_meta.source_frame_width,  1) < 0.95 and
        0.05 < cy / max(frame_meta.source_frame_height, 1) < 0.95
    )

    return (conf_score * 0.45 +
            size_score * 0.35 +
            aspect_ok  * 0.10 +
            edge_ok    * 0.10)


# ──────────────────────────────────────────────────────────────────────────────
# CROP EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_crop_bytes(gst_buffer, frame_meta, rect):
    """
    Map NvBufSurface → numpy array, crop the bbox, return JPEG bytes.
    The surface pointer is only valid inside this probe call — copy immediately.
    """
    try:
        n_frame = pyds.get_nvds_buf_surface(
            hash(gst_buffer), frame_meta.batch_id
        )
        frame_bgr = np.array(n_frame, copy=True)

        fh, fw = frame_bgr.shape[:2]
        x1 = max(0,  int(rect.left))
        y1 = max(0,  int(rect.top))
        x2 = min(fw, int(rect.left + rect.width))
        y2 = min(fh, int(rect.top  + rect.height))

        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame_bgr[y1:y2, x1:x2]
        _, crop_buf = cv2.imencode(".jpg", crop,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
        _, frame_buf = cv2.imencode(".jpg", frame_bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        return crop_buf.tobytes(), frame_buf.tobytes()

    except Exception as e:
        print(f"[WARN] extract_crop_bytes: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# FLUSH & SELECT
# ──────────────────────────────────────────────────────────────────────────────

def flush_and_select(key):
    """
    Sort the frame buffer for `key` by quality score, write top-N to disk,
    then release the buffer.
    """
    src_id, oid = key
    frames = object_frame_buffer.get(key, [])

    if not frames:
        object_frame_buffer.pop(key, None)
        last_seen.pop(key, None)
        return

    sorted_frames = sorted(frames, key=lambda x: x["score"], reverse=True)

    frames_dir = os.path.join(OUTPUT_FRAMES_DIR, f"src{src_id}")
    crops_dir  = os.path.join(OUTPUT_CROPS_DIR,  f"src{src_id}")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(crops_dir,  exist_ok=True)

    # Top-N full frames
    for rank, entry in enumerate(sorted_frames[:TOP_FRAMES], 1):
        path = os.path.join(
            frames_dir,
            f"obj{oid}_rank{rank}_f{entry['frame_num']}_s{entry['score']:.2f}.jpg"
        )
        with open(path, "wb") as fh:
            fh.write(entry["frame"])

    # Top-N crops
    for rank, entry in enumerate(sorted_frames[:TOP_CROPS], 1):
        path = os.path.join(
            crops_dir,
            f"obj{oid}_crop{rank}_f{entry['frame_num']}_s{entry['score']:.2f}.jpg"
        )
        with open(path, "wb") as fh:
            fh.write(entry["crop"])

    print(f"[FLUSH] src={src_id} obj={oid}  "
          f"frames={min(TOP_FRAMES, len(sorted_frames))}  "
          f"crops={min(TOP_CROPS, len(sorted_frames))}  "
          f"best_score={sorted_frames[0]['score']:.3f}")

    del object_frame_buffer[key]
    last_seen.pop(key, None)


# ──────────────────────────────────────────────────────────────────────────────
# TERMINATION DETECTION  (dual-channel)
# ──────────────────────────────────────────────────────────────────────────────

def get_active_ids(batch_meta):
    """All (src_id, object_id) tuples present in NvDsObjectMeta this batch."""
    active = set()
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                active.add((frame_meta.source_id, obj.object_id))
            except StopIteration:
                break
            l_obj = l_obj.next
        l_frame = l_frame.next
    return active


def get_shadow_ids(batch_meta):
    """All (src_id, object_id) tuples present in NvDsPastFrameObjBatch this batch."""
    shadow = set()
    l_usr = batch_meta.batch_user_meta_list
    while l_usr:
        try:
            usr_meta = pyds.NvDsUserMeta.cast(l_usr.data)
        except StopIteration:
            break
        if (usr_meta.base_meta.meta_type ==
                pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            try:
                pf_batch = pyds.NvDsPastFrameObjBatch.cast(
                    usr_meta.user_meta_data
                )
                for s in range(pf_batch.numAllocated):
                    stream = pf_batch.list[s]
                    for o in range(stream.numFilled):
                        obj = stream.list[o]
                        shadow.add((stream.streamID, obj.uniqueId))
            except Exception as e:
                print(f"[WARN] get_shadow_ids: {e}")
        try:
            l_usr = l_usr.next
        except StopIteration:
            break
    return shadow


def find_terminated(batch_meta):
    """
    Returns set of (src_id, object_id) that just terminated this batch.
    Mutates module-level known_ids for the next call.
    """
    active      = get_active_ids(batch_meta)
    shadow      = get_shadow_ids(batch_meta)
    still_alive = active | shadow
    terminated  = known_ids - still_alive   # was known, now gone from both

    known_ids.clear()
    known_ids.update(still_alive)
    return terminated


# ──────────────────────────────────────────────────────────────────────────────
# SAFETY-NET TIMEOUT FLUSH
# ──────────────────────────────────────────────────────────────────────────────

def flush_stale_tracks(current_frame_num):
    """
    Flush any track not seen for TIMEOUT_FRAMES.
    Handles objects still visible at EOS and very long shadow windows.
    """
    stale = [
        k for k, f in last_seen.items()
        if current_frame_num - f > TIMEOUT_FRAMES
    ]
    for key in stale:
        flush_and_select(key)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROBE
# ──────────────────────────────────────────────────────────────────────────────

def buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    current_frame_num = 0

    # ── 1. Accumulate live detections ────────────────────────────────────────
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        src_id        = frame_meta.source_id
        frame_num     = frame_meta.frame_num
        current_frame_num = max(current_frame_num, frame_num)

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            key  = (src_id, obj_meta.object_id)
            rect = obj_meta.rect_params

            score               = frame_quality_score(
                                      obj_meta.confidence, rect, frame_meta)
            crop_bytes, frame_bytes = extract_crop_bytes(
                                          gst_buffer, frame_meta, rect)

            if crop_bytes is not None:
                buf = object_frame_buffer[key]
                buf.append({
                    "score":     score,
                    "frame_num": frame_num,
                    "bbox":      (rect.left, rect.top,
                                  rect.width, rect.height),
                    "crop":      crop_bytes,
                    "frame":     frame_bytes,
                })

                # Cap buffer — evict lowest-score entry on overflow
                if len(buf) > BUFFER_CAP:
                    buf.sort(key=lambda x: x["score"], reverse=True)
                    buf.pop()

            last_seen[key] = frame_num

            l_obj = l_obj.next
        l_frame = l_frame.next

    # ── 2. Dual-channel termination detection ────────────────────────────────
    terminated = find_terminated(batch_meta)
    for key in terminated:
        if key in object_frame_buffer:
            flush_and_select(key)

    # ── 3. Safety-net timeout flush ──────────────────────────────────────────
    if current_frame_num > 0:
        flush_stale_tracks(current_frame_num)

    return Gst.PadProbeReturn.OK


# ──────────────────────────────────────────────────────────────────────────────
# EOS HANDLER  — flush everything remaining
# ──────────────────────────────────────────────────────────────────────────────

def eos_flush():
    """Called on EOS to flush all remaining buffered tracks."""
    remaining = list(object_frame_buffer.keys())
    print(f"[EOS] flushing {len(remaining)} remaining tracks")
    for key in remaining:
        flush_and_select(key)


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(input_path, output_path):
    """
    GStreamer pipeline:

      filesrc → qtdemux → h264parse → nvv4l2decoder
             → nvstreammux
             → nvinfer (PGIE)
             → nvtracker (NvDCF)
             → nvdsosd
             → nvvideoconvert
             → x264enc
             → mp4mux
             → filesink
    """
    pipeline = Gst.Pipeline.new("best-frame-pipeline")

    def make(factory, name):
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Failed to create element: {factory}")
        pipeline.add(el)
        return el

    # Source
    filesrc    = make("filesrc",       "filesrc")
    qtdemux    = make("qtdemux",       "qtdemux")
    h264parse  = make("h264parse",     "h264parse")
    decoder    = make("nvv4l2decoder", "decoder")

    # Mux
    streammux  = make("nvstreammux",   "streammux")

    # Inference + tracking
    pgie       = make("nvinfer",       "pgie")
    tracker    = make("nvtracker",     "tracker")

    # OSD
    vidconv1   = make("nvvideoconvert","vidconv1")
    osd        = make("nvdsosd",       "osd")

    # Encode + sink
    vidconv2   = make("nvvideoconvert","vidconv2")
    capsfilter = make("capsfilter",    "capsfilter")
    encoder    = make("x264enc",       "encoder")
    h264parse2 = make("h264parse",     "h264parse2")
    muxer      = make("mp4mux",        "muxer")
    filesink   = make("filesink",      "filesink")

    # ── Element properties ───────────────────────────────────────────────────

    filesrc.set_property("location", input_path)

    streammux.set_property("batch-size",         1)
    streammux.set_property("width",              1920)
    streammux.set_property("height",             1080)
    streammux.set_property("batched-push-timeout", 4000000)
    streammux.set_property("live-source",        False)

    pgie.set_property("config-file-path", PGIE_CONFIG)

    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("tracker-width",  640)
    tracker.set_property("tracker-height", 384)

    osd.set_property("process-mode",  0)   # CPU mode
    osd.set_property("display-text",  True)
    osd.set_property("display-bbox",  True)

    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    capsfilter.set_property("caps", caps)

    encoder.set_property("bitrate",     4000)
    encoder.set_property("speed-preset", "fast")
    encoder.set_property("key-int-max",  30)

    filesink.set_property("location",   output_path)
    filesink.set_property("sync",       False)

    # ── Link static elements ─────────────────────────────────────────────────

    filesrc.link(qtdemux)
    # qtdemux → h264parse linked dynamically via pad-added
    h264parse.link(decoder)

    # decoder → streammux  (request pad)
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad  = decoder.get_static_pad("src")
    if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
        raise RuntimeError("decoder → streammux link failed")

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(vidconv1)
    vidconv1.link(osd)
    osd.link(vidconv2)
    vidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(h264parse2)
    h264parse2.link(muxer)
    muxer.link(filesink)

    # ── Dynamic pad: qtdemux → h264parse ────────────────────────────────────
    def on_pad_added(element, pad):
        caps_str = pad.get_current_caps().to_string() if pad.get_current_caps() else ""
        if "video" in caps_str or "h264" in caps_str or not caps_str:
            sink = h264parse.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

    qtdemux.connect("pad-added", on_pad_added)

    # ── Probe on OSD sink pad ────────────────────────────────────────────────
    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        raise RuntimeError("Cannot get OSD sink pad")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe, 0)

    return pipeline


# ──────────────────────────────────────────────────────────────────────────────
# BUS MESSAGE HANDLER
# ──────────────────────────────────────────────────────────────────────────────

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] End of stream")
        eos_flush()
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[BUS] Warning: {err} | {debug}")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[BUS] Error: {err} | {debug}")
        loop.quit()
    return True


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepStream best-frame selection pipeline"
    )
    parser.add_argument("--input",  required=True, help="Input MP4 path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CROPS_DIR,  exist_ok=True)

    Gst.init(None)

    print(f"[INIT] Input  : {args.input}")
    print(f"[INIT] Output : {args.output}")
    print(f"[INIT] Frames → {OUTPUT_FRAMES_DIR}/")
    print(f"[INIT] Crops  → {OUTPUT_CROPS_DIR}/")

    pipeline = build_pipeline(args.input, args.output)

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    print("[PIPELINE] Running ...")

    try:
        loop.run()
    except KeyboardInterrupt:
        print("[PIPELINE] Interrupted — flushing remaining tracks ...")
        eos_flush()

    pipeline.set_state(Gst.State.NULL)
    print("[PIPELINE] Done.")
    print(f"  Annotated video : {args.output}")
    print(f"  Best frames     : {OUTPUT_FRAMES_DIR}/")
    print(f"  Best crops      : {OUTPUT_CROPS_DIR}/")


if __name__ == "__main__":
    main()
