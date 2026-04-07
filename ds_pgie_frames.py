"""
DeepStream PGIE Detection Pipeline — RGBA Frame + Metadata Storage
==================================================================
Reads from an RTSP stream (or MP4 file), runs PGIE detection, and saves
per-frame RGBA PNGs + JSON metadata (bounding boxes, classes, confidence)
to a local output folder.

Directory layout after a run:
  output/
    frames/
      frame_000001_src0.png        ← RGBA PNG with boxes drawn
      frame_000001_src0.json       ← bounding-box metadata
      frame_000002_src0.png
      frame_000002_src0.json
      ...
    run_manifest.jsonl             ← one JSON line per saved frame (streaming)

Requirements:
  - NVIDIA DeepStream 8.x Python bindings (pyds)
  - GStreamer 1.x Python bindings (gi / PyGObject)
  - OpenCV (cv2) with Python bindings
  - numpy

Usage:
  python pipeline.py --input rtsp://192.168.1.10/stream1 --config pgie_config.txt
  python pipeline.py --input file:///data/test.mp4      --config pgie_config.txt
  python pipeline.py --input rtsp://...  --config pgie_config.txt --save-every 5
"""

import argparse
import ctypes
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

import pyds


# ─────────────────────────────────────────────────────────────────────────────
# Constants / Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR   = "./output"
DEFAULT_SAVE_EVERY   = 1          # save every N frames (1 = save all)
DEFAULT_CONFIDENCE   = 0.30       # minimum detection confidence to keep
DEFAULT_MUXER_W      = 1920
DEFAULT_MUXER_H      = 1080
DEFAULT_MUXER_FPS    = 30         # used for live-source timing

# Colour palette for bounding-box overlays (BGR → used by OpenCV)
# Index matches your PGIE class id
CLASS_COLORS_BGR = [
    (0,   255, 100),   # 0 – e.g. car
    (0,   128, 255),   # 1 – e.g. truck
    (255, 50,  50 ),   # 2 – e.g. person
    (200, 0,   255),   # 3 – e.g. bus
    (255, 200, 0  ),   # 4 – e.g. motorcycle
    (0,   220, 220),   # 5
    (255, 100, 200),   # 6
    (100, 255, 0  ),   # 7
]

# Map class id → human-readable label (override / extend as needed)
CLASS_LABELS: dict[int, str] = {
    0: "car",
    1: "truck",
    2: "person",
    3: "bus",
    4: "motorcycle",
    5: "bicycle",
    6: "van",
    7: "unknown",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def class_label(class_id: int) -> str:
    return CLASS_LABELS.get(class_id, f"class_{class_id}")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]


def make_dirs(output_dir: str):
    frames_dir = os.path.join(output_dir, "frames")
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    return frames_dir


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction from NvBufSurface (RGBA, GPU → CPU copy)
# ─────────────────────────────────────────────────────────────────────────────

def extract_rgba_frame(frame_meta, batch_id: int) -> Optional[np.ndarray]:
    """
    Copies one surface from the NvBufSurface batch into a CPU numpy array
    in RGBA format.

    Returns:
        np.ndarray of shape (H, W, 4), dtype=uint8, in RGBA order.
        None if extraction fails.

    Notes:
        - nvvideoconvert upstream must output RGBA (configured in the pipeline).
        - pyds.get_nvds_buf_surface returns a numpy view over the mapped surface.
          We immediately copy it (.copy()) so the surface can be unmapped safely.
    """
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(frame_meta), batch_id)
        # n_frame is a numpy array of shape (H, W, 4) in RGBA
        frame_copy = np.array(n_frame, copy=True, order="C")
        return frame_copy
    except Exception as exc:
        print(f"[WARN] extract_rgba_frame failed for batch_id={batch_id}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Metadata collection from NvDsObjectMeta
# ─────────────────────────────────────────────────────────────────────────────

def collect_detections(frame_meta, min_confidence: float) -> list[dict]:
    """
    Walks the object meta list for one frame and returns a list of detection dicts.

    Each dict:
        {
            "object_id":  int,      # tracker id (-1 if no tracker in pipeline)
            "class_id":   int,
            "label":      str,
            "confidence": float,
            "bbox": {
                "left":   float,    # pixels, relative to full frame
                "top":    float,
                "width":  float,
                "height": float,
            }
        }
    """
    detections = []
    l_obj = frame_meta.obj_meta_list

    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        conf = obj_meta.confidence
        if conf >= min_confidence:
            rect = obj_meta.rect_params
            detections.append({
                "object_id":  int(obj_meta.object_id),
                "class_id":   int(obj_meta.class_id),
                "label":      obj_meta.obj_label if obj_meta.obj_label else class_label(obj_meta.class_id),
                "confidence": round(float(conf), 4),
                "bbox": {
                    "left":   round(float(rect.left),   2),
                    "top":    round(float(rect.top),    2),
                    "width":  round(float(rect.width),  2),
                    "height": round(float(rect.height), 2),
                },
            })

        try:
            l_obj = l_obj.next
        except StopIteration:
            break

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Annotation: draw boxes on the RGBA frame
# ─────────────────────────────────────────────────────────────────────────────

def annotate_rgba(rgba: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draws bounding boxes and labels on an RGBA numpy array in-place.
    Returns the annotated array (same object).

    OpenCV works in BGR(A). The frame is RGBA from DeepStream.
    We convert RGBA → BGRA for OpenCV drawing, then convert back.
    """
    # RGBA → BGRA (in-place channel swap)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    for det in detections:
        b = det["bbox"]
        x1 = int(b["left"])
        y1 = int(b["top"])
        x2 = int(b["left"] + b["width"])
        y2 = int(b["top"]  + b["height"])

        bgr  = color_for_class(det["class_id"])
        bgra_color = (*bgr, 255)   # fully opaque

        # Bounding box
        cv2.rectangle(bgra, (x1, y1), (x2, y2), bgra_color, thickness=2)

        # Label background + text
        label_text = f"{det['label']} {det['confidence']:.2f}"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Dark semi-transparent background for readability
        bg_x1, bg_y1 = x1, max(0, y1 - th - baseline - 4)
        bg_x2, bg_y2 = x1 + tw + 4, y1
        overlay = bgra.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, bgra, 0.4, 0, bgra)

        cv2.putText(bgra, label_text, (x1 + 2, y1 - baseline - 2),
                    font, font_scale, bgra_color, thickness, cv2.LINE_AA)

    # BGRA → RGBA
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


# ─────────────────────────────────────────────────────────────────────────────
# Save one frame + its metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_frame_and_metadata(
    frames_dir:  str,
    manifest_fh,
    manifest_lock: threading.Lock,
    rgba:        np.ndarray,
    detections:  list[dict],
    frame_num:   int,
    source_id:   int,
    pts:         int,
    annotate:    bool = True,
):
    """
    Saves:
      frames_dir/frame_{frame_num:06d}_src{source_id}.png   ← RGBA PNG
      frames_dir/frame_{frame_num:06d}_src{source_id}.json  ← detection metadata
    And appends one line to the run manifest.
    """
    stem = f"frame_{frame_num:06d}_src{source_id}"
    png_path  = os.path.join(frames_dir, f"{stem}.png")
    json_path = os.path.join(frames_dir, f"{stem}.json")

    # ── annotate then save PNG ────────────────────────────────────────────────
    out_frame = annotate_rgba(rgba.copy(), detections) if annotate else rgba

    # cv2.imwrite with RGBA: use imencode with PNG params to preserve alpha
    success, buf = cv2.imencode(
        ".png", cv2.cvtColor(out_frame, cv2.COLOR_RGBA2BGRA),
        [cv2.IMWRITE_PNG_COMPRESSION, 3]   # 0=no compression, 9=max; 3 is fast+decent
    )
    if success:
        with open(png_path, "wb") as f:
            f.write(buf.tobytes())
    else:
        print(f"[WARN] imencode failed for {stem}")

    # ── save JSON metadata ────────────────────────────────────────────────────
    meta = {
        "frame_number": frame_num,
        "source_id":    source_id,
        "pts_ns":       pts,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "image_file":   os.path.basename(png_path),
        "frame_size":   {"width": rgba.shape[1], "height": rgba.shape[0]},
        "num_detections": len(detections),
        "detections":   detections,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── append to manifest ────────────────────────────────────────────────────
    manifest_entry = {
        "frame_number":   frame_num,
        "source_id":      source_id,
        "png":            png_path,
        "json":           json_path,
        "num_detections": len(detections),
    }
    with manifest_lock:
        manifest_fh.write(json.dumps(manifest_entry) + "\n")
        manifest_fh.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Pad probe — the core callback
# ─────────────────────────────────────────────────────────────────────────────

def make_probe(args, frames_dir, manifest_fh, manifest_lock):
    """
    Returns a GStreamer sink-pad probe function bound to the given config.

    The probe is attached to the sink pad of nvdsosd (or tee/fakesink if no OSD).
    It fires on every batch buffer, extracts frames + metadata, and saves them.
    """
    save_every     = args.save_every
    min_confidence = args.min_confidence
    annotate       = not args.no_annotate

    frame_counter: dict[int, int] = {}   # source_id → local frame count

    def probe_fn(pad, info):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        batch_id = 0   # surface index within the batch

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id  = frame_meta.source_id
            frame_num  = frame_meta.frame_num
            pts        = gst_buffer.pts

            # ── per-source save-every counter ──────────────────────────────
            cnt = frame_counter.get(source_id, 0)
            frame_counter[source_id] = cnt + 1

            if cnt % save_every == 0:
                # ── collect detections from metadata ───────────────────────
                detections = collect_detections(frame_meta, min_confidence)

                # ── extract RGBA pixels from GPU surface ───────────────────
                rgba = extract_rgba_frame(frame_meta, batch_id)

                if rgba is not None:
                    save_frame_and_metadata(
                        frames_dir=frames_dir,
                        manifest_fh=manifest_fh,
                        manifest_lock=manifest_lock,
                        rgba=rgba,
                        detections=detections,
                        frame_num=frame_num,
                        source_id=source_id,
                        pts=pts,
                        annotate=annotate,
                    )

                    n_det = len(detections)
                    print(
                        f"[frame {frame_num:06d}][src {source_id}] "
                        f"saved — {n_det} detection(s)"
                        + (f": {[d['label'] for d in detections]}" if n_det else "")
                    )

            batch_id += 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return probe_fn


# ─────────────────────────────────────────────────────────────────────────────
# GStreamer Bus message handler
# ─────────────────────────────────────────────────────────────────────────────

def bus_call(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.EOS:
        print("[INFO] End-of-stream received.")
        loop.quit()
    elif mtype == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f"[WARN] {err.message}  [{debug}]")
    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err.message}  [{debug}]")
        loop.quit()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(args) -> Gst.Pipeline:
    """
    Constructs and links the GStreamer pipeline:

    For RTSP:
      rtspsrc → rtph264depay → h264parse → nvv4l2decoder
          → nvstreammux → nvinfer (PGIE)
          → nvvideoconvert (→ RGBA)
          → nvdsosd
          → nvvideoconvert (→ RGBA for probe)
          → appsink / fakesink

    For file (MP4/H264):
      filesrc → qtdemux → h264parse → nvv4l2decoder
          → nvstreammux → nvinfer (PGIE)
          → nvvideoconvert (→ RGBA)
          → nvdsosd
          → nvvideoconvert (→ RGBA for probe)
          → fakesink

    The pad probe is added on the sink pad of the final nvvideoconvert
    so we read RGBA frames after OSD drawing has been completed.
    """
    is_rtsp = args.input.startswith("rtsp://") or args.input.startswith("rtsps://")
    is_file = args.input.startswith("file://") or args.input.endswith((".mp4", ".h264", ".mkv"))

    pipeline = Gst.Pipeline.new("ds-pgie-frame-store")

    # ── Source elements ───────────────────────────────────────────────────────
    if is_rtsp:
        source   = Gst.ElementFactory.make("rtspsrc",      "src")
        depay    = Gst.ElementFactory.make("rtph264depay", "depay")
        parser   = Gst.ElementFactory.make("h264parse",    "parser")

        source.set_property("location",         args.input)
        source.set_property("protocols",        "tcp")          # avoid UDP loss
        source.set_property("latency",          200)
        source.set_property("drop-on-latency",  True)
        source.set_property("buffer-mode",      4)

        for el in (source, depay, parser):
            pipeline.add(el)

        # rtspsrc pad is dynamic — link depay when pad appears
        def on_pad_added(src_el, new_pad):
            sink_pad = depay.get_static_pad("sink")
            if not sink_pad.is_linked():
                caps = new_pad.get_current_caps()
                if caps and "video" in caps.to_string():
                    new_pad.link(sink_pad)

        source.connect("pad-added", on_pad_added)
        depay.link(parser)
        last_before_decoder = parser

    else:
        filesrc  = Gst.ElementFactory.make("filesrc",   "src")
        qtdemux  = Gst.ElementFactory.make("qtdemux",   "demux")
        parser   = Gst.ElementFactory.make("h264parse", "parser")

        uri = args.input.replace("file://", "")
        filesrc.set_property("location", uri)

        for el in (filesrc, qtdemux, parser):
            pipeline.add(el)

        filesrc.link(qtdemux)

        def on_demux_pad(demux, pad):
            if pad.get_name().startswith("video"):
                pad.link(parser.get_static_pad("sink"))

        qtdemux.connect("pad-added", on_demux_pad)
        last_before_decoder = parser

    # ── Decoder ───────────────────────────────────────────────────────────────
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    decoder.set_property("enable-max-performance", 1)
    decoder.set_property("num-extra-surfaces",     4)
    decoder.set_property("skip-frames",            0)
    pipeline.add(decoder)
    last_before_decoder.link(decoder)

    # ── Streammux ─────────────────────────────────────────────────────────────
    streammux = Gst.ElementFactory.make("nvstreammux", "mux")
    streammux.set_property("width",                args.width)
    streammux.set_property("height",               args.height)
    streammux.set_property("batch-size",           1)
    streammux.set_property("live-source",          1 if is_rtsp else 0)
    streammux.set_property("batched-push-timeout", 33333)  # 33ms ≈ 30fps
    streammux.set_property("enable-padding",       0)
    pipeline.add(streammux)

    # Link decoder → streammux sink pad
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad  = decoder.get_static_pad("src")
    srcpad.link(sinkpad)

    # ── PGIE (nvinfer) ────────────────────────────────────────────────────────
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", args.config)
    pipeline.add(pgie)
    streammux.link(pgie)

    # ── nvvideoconvert #1 → RGBA (before OSD) ─────────────────────────────────
    nvconv1 = Gst.ElementFactory.make("nvvideoconvert", "conv1")
    pipeline.add(nvconv1)
    pgie.link(nvconv1)

    # Set caps to RGBA after first converter
    caps_rgba = Gst.caps_from_string("video/x-raw(memory:NVMM),format=RGBA")
    caps_filter1 = Gst.ElementFactory.make("capsfilter", "cf1")
    caps_filter1.set_property("caps", caps_rgba)
    pipeline.add(caps_filter1)
    nvconv1.link(caps_filter1)

    # ── nvdsosd — draws metadata overlays on the frame ────────────────────────
    osd = Gst.ElementFactory.make("nvdsosd", "osd")
    osd.set_property("process-mode", 1)   # 1=GPU, 0=CPU
    osd.set_property("display-text", 1)
    pipeline.add(osd)
    caps_filter1.link(osd)

    # ── nvvideoconvert #2 → RGBA (after OSD, for probe pixel access) ──────────
    nvconv2 = Gst.ElementFactory.make("nvvideoconvert", "conv2")
    pipeline.add(nvconv2)
    osd.link(nvconv2)

    caps_filter2 = Gst.ElementFactory.make("capsfilter", "cf2")
    caps_filter2.set_property("caps", caps_rgba)
    pipeline.add(caps_filter2)
    nvconv2.link(caps_filter2)

    # ── Sink ──────────────────────────────────────────────────────────────────
    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", 0)          # no A/V sync — process as fast as possible
    sink.set_property("async", 0)
    sink.set_property("enable-last-sample", 0)
    pipeline.add(sink)
    caps_filter2.link(sink)

    return pipeline, caps_filter2   # return filter2 for probe attachment


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepStream PGIE pipeline — saves RGBA frames + JSON metadata per detection frame."
    )
    p.add_argument("--input",          required=True,
                   help="RTSP URL (rtsp://...) or file path / file:// URI")
    p.add_argument("--config",         required=True,
                   help="Path to PGIE nvinfer config file (.txt)")
    p.add_argument("--output-dir",     default=DEFAULT_OUTPUT_DIR,
                   help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--save-every",     type=int,   default=DEFAULT_SAVE_EVERY,
                   help="Save every N frames per source (default: 1 = save all)")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help=f"Minimum detection confidence to include (default: {DEFAULT_CONFIDENCE})")
    p.add_argument("--width",          type=int,   default=DEFAULT_MUXER_W,
                   help=f"Muxer output width  (default: {DEFAULT_MUXER_W})")
    p.add_argument("--height",         type=int,   default=DEFAULT_MUXER_H,
                   help=f"Muxer output height (default: {DEFAULT_MUXER_H})")
    p.add_argument("--no-annotate",    action="store_true",
                   help="Save clean frames without drawn boxes")
    p.add_argument("--max-frames",     type=int,   default=0,
                   help="Stop after saving this many frames (0 = unlimited)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Output directories ────────────────────────────────────────────────────
    frames_dir = make_dirs(args.output_dir)
    manifest_path = os.path.join(args.output_dir, "run_manifest.jsonl")

    print(f"[INFO] Output dir : {os.path.abspath(args.output_dir)}")
    print(f"[INFO] Frames dir : {os.path.abspath(frames_dir)}")
    print(f"[INFO] Manifest   : {os.path.abspath(manifest_path)}")
    print(f"[INFO] Save every : {args.save_every} frame(s)")
    print(f"[INFO] Min conf   : {args.min_confidence}")

    # ── Initialise GStreamer + DeepStream ─────────────────────────────────────
    Gst.init(None)

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline, probe_element = build_pipeline(args)

    # ── Attach probe ──────────────────────────────────────────────────────────
    manifest_fh   = open(manifest_path, "a")
    manifest_lock = threading.Lock()

    probe_fn = make_probe(args, frames_dir, manifest_fh, manifest_lock)

    sinkpad = probe_element.get_static_pad("sink")
    sinkpad.add_probe(Gst.PadProbeType.BUFFER, probe_fn)

    # ── Bus watch ─────────────────────────────────────────────────────────────
    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # ── Start ─────────────────────────────────────────────────────────────────
    print("[INFO] Setting pipeline to PLAYING ...")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[ERROR] Unable to set pipeline to PLAYING state.")
        sys.exit(1)

    print("[INFO] Pipeline running. Press Ctrl+C to stop.\n")

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Stopping pipeline ...")
        pipeline.set_state(Gst.State.NULL)
        manifest_fh.close()
        print(f"[INFO] Done. Frames saved to: {os.path.abspath(frames_dir)}")


if __name__ == "__main__":
    main()
