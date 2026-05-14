# DeepStream VSS Pipeline Architecture

Production-grade architecture for multi-camera video analytics with Detection, Tracking, ReID, License Plate OCR, and top-frame/crop extraction — all metadata flowing to Kafka.

---

## Design Principles

- **DeepStream owns what only DeepStream can do**: GPU inference, frame-synchronized tracking, raw pixel access
- **One probe, fire-and-forget**: the pipeline thread never blocks on I/O or encoding
- **Thin pipeline, smart consumers**: detection and tracking in DeepStream; ReID, OCR, aggregation downstream
- **Crops never go through nvmsgconv**: binary/image data takes the custom producer path
- **Bounded queues with drop policy**: lose crops before losing frames

---

## Full Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                          VIDEO SOURCES (N cameras)                                   ║
║         [RTSP/File cam-0]  [RTSP/File cam-1]  ...  [RTSP/File cam-N]               ║
╚══════════════════╦═════════════════════════════════════════════════════════════════════╝
                   ║  uridecodebin × N
                   ▼
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        DEEPSTREAM PIPELINE  (GPU thread)                             ║
║                                                                                      ║
║   [nvstreammux]                                                                      ║
║        │  batched frames                                                             ║
║        ▼                                                                             ║
║   [nvinfer]  ─── Primary Detector                                                   ║
║        │         (person / vehicle / face)                                           ║
║        ▼                                                                             ║
║   [nvtracker]  ── NvDCF / ByteTrack                                                 ║
║        │          assigns track_id, maintains state                                  ║
║        ▼                                                                             ║
║   [nvinfer]  ─── Secondary Detector                                                 ║
║        │         LP bbox only  (no OCR here)                                        ║
║        │                                                                             ║
║        ▼                                                                             ║
║   ┌─────────────────────────────────────────────────────┐                           ║
║   │         UNIFIED SINK PAD PROBE  (one probe only)    │                           ║
║   │                                                      │                           ║
║   │  for each obj_meta in batch:                        │                           ║
║   │    • compute quality_score  (fast heuristic)        │                           ║
║   │    • push CropTask → ASYNC QUEUE  (non-blocking)    │                           ║
║   │    • attach NvDsEventMsgMeta  (for nvmsgconv)       │                           ║
║   │    • emit TRACK_NEW / TRACK_LOST signals             │                           ║
║   │      → push TrackEvent → ASYNC QUEUE                │                           ║
║   └───────────────┬──────────────────────────────────────┘                          ║
║                   │  returns immediately                                             ║
║                   ▼                                                                  ║
║   [nvmsgconv]  +  [nvmsgbroker]                                                     ║
║        │          libnvds_kafka_proto.so                                             ║
║        │          NvDsEventMsgMeta → JSON                                           ║
╚════════╬═════════════════════════════════════════════════════════════════════════════╝
         │
         │ (sync, low-latency path)
         ▼
topic: vss.detections
  { stream_id, frame_num, ts, track_id, class,
    bbox, confidence, pipeline_instance_id }


╔══════════════════════════════════════════════════════════════════════════════════════╗
║                     ASYNC LAYER  (separate thread pool, same process)               ║
║                                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────┐           ║
║   │  ASYNC QUEUE  (bounded, e.g. maxsize=2000, drop-on-full policy)    │           ║
║   │                                                                      │           ║
║   │   CropTask  { surface_ref, obj_meta_copy, frame_meta_copy }        │           ║
║   │   TrackEvent{ track_id, event_type, stream_id, ts }                │           ║
║   └──────────────────────┬───────────────────────────────────────────── ┘           ║
║                           │                                                          ║
║            ┌──────────────┴──────────────────────┐                                 ║
║            ▼                                      ▼                                 ║
║   ┌─────────────────────┐             ┌────────────────────────┐                   ║
║   │  CROP WORKER POOL   │             │  TRACK EVENT WORKER    │                   ║
║   │  (N threads / async)│             │  (single thread, stateful)                 │
║   │                     │             │                        │                   ║
║   │ • GPU→CPU copy      │             │ • TRACK_NEW            │                   ║
║   │ • JPEG encode       │             │   → init track buffer  │                   ║
║   │ • route by class:   │             │ • TRACK_UPDATE         │                   ║
║   │   person → crops    │             │   → update last seen   │                   ║
║   │   vehicle → crops   │             │ • TRACK_LOST           │                   ║
║   │   LP region → lp    │             │   → flush top-1/top-K  │                   ║
║   └──────────┬──────────┘             └────────────┬───────────┘                   ║
╚══════════════╬═══════════════════════════════════════╬═══════════════════════════════╝
               │                                       │
    ┌──────────┴───────────┐               ┌───────────┴────────────┐
    ▼                      ▼               ▼                        ▼
topic:              topic:           topic:                   topic:
vss.objects.crops   vss.lp.crops     vss.tracks.events        vss.tracks.summary
                                     (on TRACK_LOST only)


╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              KAFKA  (topics)                                         ║
║                                                                                      ║
║  Ingest topics                                                                       ║
║   vss.detections       vss.objects.crops    vss.lp.crops                            ║
║   vss.tracks.events    vss.tracks.summary                                           ║
║                                                                                      ║
║  Output topics                                                                       ║
║   vss.reid.matches     vss.lp.results       vss.alerts                              ║
╚══════════╦══════════════════════╦═══════════════════════╦════════════════════════════╝
           │                      │                        │
     ┌─────┘              ┌───────┘                ┌───────┘
     ▼                    ▼                         ▼

╔═══════════════╗  ╔═══════════════════╗  ╔══════════════════════════════════════╗
║  REID         ║  ║  LP OCR CONSUMER  ║  ║  TRACK AGGREGATOR CONSUMER          ║
║  CONSUMER     ║  ║                   ║  ║                                      ║
║               ║  ║ in: vss.lp.crops  ║  ║ in: vss.objects.crops               ║
║ in:           ║  ║                   ║  ║     vss.tracks.events                ║
║ vss.objects   ║  ║ • PaddleOCR       ║  ║     vss.tracks.summary               ║
║ .crops        ║  ║   or TRT-OCR      ║  ║                                      ║
║               ║  ║ • normalize plate ║  ║ • accumulate crops per track_id      ║
║ • extract     ║  ║   string          ║  ║   in bounded heap (maxsize=50)       ║
║   embeddings  ║  ║ • confidence      ║  ║ • rank by quality_score              ║
║   (CLIP/OSNet)║  ║   threshold       ║  ║ • on TRACK_LOST:                     ║
║ • FAISS/Redis ║  ║                   ║  ║   emit top-1 frame + top-K crops     ║
║   gallery     ║  ║ out:              ║  ║   to vss.tracks.summary              ║
║   per zone    ║  ║ vss.lp.results    ║  ║                                      ║
║               ║  ║ { plate_str,      ║  ║ state: Redis hash                    ║
║ • cross-cam   ║  ║   confidence,     ║  ║   key: track_id                      ║
║   match       ║  ║   track_id,       ║  ║   val: heap of (score, s3_key)       ║
║               ║  ║   stream_id, ts } ║  ║                                      ║
║ out:          ║  ╚═══════════════════╝  ╚══════════════════════════════════════╝
║ vss.reid      ║
║ .matches      ║  ╔═══════════════════════════════════════════════════════════════╗
║ { person_id,  ║  ║  ANALYTICS / STORAGE CONSUMER                                ║
║   track_ids[] ║  ║                                                               ║
║   streams[]   ║  ║  in: vss.detections + vss.tracks.summary + vss.reid.matches  ║
║   confidence  ║  ║      + vss.lp.results                                         ║
║   ts }        ║  ║                                                               ║
╚═══════════════╝  ║  • write to TimescaleDB / Elasticsearch / S3                 ║
                   ║  • trigger alerts (dwell time, zone crossing, LP watchlist)   ║
                   ║  • feed dashboard (Grafana / custom UI)                       ║
                   ╚═══════════════════════════════════════════════════════════════╝
```

---

## Task Split: Pipeline vs Consumer

| Feature | Where | Reason |
|---|---|---|
| Primary Detection | DeepStream (nvinfer) | Batched GPU inference; feeds tracker |
| Tracking | DeepStream (nvtracker) | Frame-synchronous; needs consecutive frames |
| LP Detection (bbox) | DeepStream (secondary nvinfer) | Needs primary bbox context; GPU-batched |
| LP OCR | LP OCR Consumer | Text recognition is latency-tolerant; scale independently |
| ReID | ReID Consumer | Needs gallery state; latency-tolerant |
| Top frame per track | Track Aggregator Consumer | Score and pick from buffered crops on TRACK_LOST |
| Top-K crops per track | Track Aggregator Consumer | Accumulate, rank, emit on TRACK_LOST |
| Track lifecycle events | Track Aggregator Consumer | Stateful aggregation on Kafka-compacted topic |
| Alerts / storage | Analytics Consumer | Terminal output; no GPU needed |

---

## Kafka Topics

### Ingest Topics (written by pipeline)

| Topic | Producer | Key | Payload |
|---|---|---|---|
| `vss.detections` | nvmsgbroker | `stream_id:frame_num` | `{ stream_id, frame_num, ts, track_id, class, bbox, confidence, pipeline_instance_id }` |
| `vss.objects.crops` | custom librdkafka | `track_id` | `{ stream_id, frame_num, ts, track_id, class, bbox, quality_score, crop_jpeg \| s3_key }` |
| `vss.lp.crops` | custom librdkafka | `track_id` | `{ lp_crop_jpeg \| s3_key, track_id, stream_id, ts }` |
| `vss.tracks.events` | async track worker | `track_id` | `{ stream_id, track_id, event: NEW\|UPDATE\|LOST, ts, bbox }` |
| `vss.tracks.summary` | async track worker | `track_id` | `{ track_id, stream_id, duration_ms, top_1_crop, top_k_crops[], first_seen, last_seen }` |

### Output Topics (written by consumers)

| Topic | Producer | Payload |
|---|---|---|
| `vss.reid.matches` | ReID Consumer | `{ person_id, track_ids[], streams[], confidence, ts }` |
| `vss.lp.results` | LP OCR Consumer | `{ plate_str, confidence, track_id, stream_id, ts }` |
| `vss.alerts` | Analytics Consumer | `{ alert_type, track_id, stream_id, detail, ts }` |

---

## Probe Implementation

One probe on the sink pad of the secondary detector. Returns immediately — no blocking I/O.

```python
def unified_probe(pad, info, u_data):
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(info.get_buffer())

    for frame_meta in nvds_frame_meta_list(batch_meta):
        for obj_meta in nvds_obj_meta_list(frame_meta):

            quality_score = (
                obj_meta.confidence
                * min(obj_meta.rect_params.width * obj_meta.rect_params.height, 1.0)
            )

            # Push crop task to bounded async queue (non-blocking)
            try:
                crop_queue.put_nowait(CropTask(
                    surface_ref  = get_surface_ref(info, frame_meta),
                    obj_meta     = copy_obj_meta(obj_meta),
                    frame_meta   = copy_frame_meta(frame_meta),
                    quality_score= quality_score,
                ))
            except asyncio.QueueFull:
                metrics.increment("crop_queue.dropped")

            # Attach NvDsEventMsgMeta for nvmsgconv (metadata only, no pixels)
            attach_event_msg_meta(batch_meta, frame_meta, obj_meta)

            # Emit track lifecycle events
            if is_new_track(obj_meta.object_id):
                track_queue.put_nowait(TrackEvent(
                    track_id   = obj_meta.object_id,
                    event_type = "NEW",
                    stream_id  = frame_meta.source_id,
                    ts         = frame_meta.ntp_timestamp,
                ))
            elif is_lost_track(obj_meta.object_id):
                track_queue.put_nowait(TrackEvent(
                    track_id   = obj_meta.object_id,
                    event_type = "LOST",
                    stream_id  = frame_meta.source_id,
                    ts         = frame_meta.ntp_timestamp,
                ))

    return Gst.PadProbeReturn.OK
```

---

## Async Crop Worker

```python
async def crop_worker(crop_queue: asyncio.Queue, producer: AIOKafkaProducer):
    while True:
        task: CropTask = await crop_queue.get()

        try:
            # GPU → CPU transfer + JPEG encode (only blocking work)
            jpeg_bytes = await asyncio.to_thread(
                extract_and_encode_crop,
                task.surface_ref,
                task.obj_meta,
            )

            s3_key = None
            if len(jpeg_bytes) > INLINE_CROP_THRESHOLD:
                s3_key = await upload_to_s3(jpeg_bytes, task)
                jpeg_bytes = None

            topic = "vss.lp.crops" if task.obj_meta.class_id == LP_CLASS else "vss.objects.crops"

            await producer.send(topic, value={
                "stream_id"    : task.frame_meta.source_id,
                "frame_num"    : task.frame_meta.frame_num,
                "ts"           : task.frame_meta.ntp_timestamp,
                "track_id"     : task.obj_meta.object_id,
                "class_id"     : task.obj_meta.class_id,
                "bbox"         : extract_bbox(task.obj_meta),
                "quality_score": task.quality_score,
                "crop_jpeg"    : b64encode(jpeg_bytes) if jpeg_bytes else None,
                "s3_key"       : s3_key,
                "pipeline_id"  : PIPELINE_INSTANCE_ID,
            })

        except Exception as e:
            log.error("crop_worker failed: %s", e)
            metrics.increment("crop_worker.errors")

        finally:
            crop_queue.task_done()
```

---

## Track Aggregator Consumer (Top Frame / Top-K)

```python
class TrackAggregator:
    def __init__(self, k: int = 5, max_buffer: int = 50):
        self.k = k
        self.max_buffer = max_buffer
        self.buffers: dict[str, list] = defaultdict(list)  # track_id → heap

    def on_crop(self, msg: dict):
        tid = msg["track_id"]
        heap = self.buffers[tid]
        heapq.heappush(heap, (msg["quality_score"], msg["crop_jpeg"] or msg["s3_key"], msg["ts"]))
        if len(heap) > self.max_buffer:
            heapq.heappop(heap)  # drop lowest-quality

    def on_track_lost(self, event: dict):
        tid = event["track_id"]
        crops = sorted(self.buffers.pop(tid, []), reverse=True)

        if not crops:
            return

        self.kafka_producer.send("vss.tracks.summary", {
            "track_id"   : tid,
            "stream_id"  : event["stream_id"],
            "duration_ms": event["duration_ms"],
            "top_1_crop" : crops[0],
            "top_k_crops": crops[:self.k],
            "first_seen" : event["first_seen"],
            "last_seen"  : event["ts"],
        })
```

---

## nvmsgconv + nvmsgbroker (Detections Path)

Use the NVIDIA-managed path for pure metadata — no custom code needed here.

```bash
# pipeline fragment (GStreamer / DeepStream config)
nvmsgconv config-file=msgconv_config.txt msg2p-newapi=1 ! \
nvmsgbroker proto-lib=/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so \
            conn-str="kafka-broker:9092" \
            topic="vss.detections" \
            sync=false
```

```ini
# msgconv_config.txt
[sensor0]
enable=1
type=Camera
id=camera-0
location=site-a/entrance
description=Entrance Camera

[place0]
enable=1
type=0
id=place-0
name=entrance

[analytics0]
enable=1
type=1
id=analytics-0
source=primary-detector
version=1.0
```

---

## State Management

| State | Store | Key Pattern | TTL |
|---|---|---|---|
| Track crop heap | Redis Hash | `{pipeline_id}:{track_id}` | 5 min after TRACK_LOST |
| ReID gallery embeddings | FAISS + Redis | `gallery:{zone_id}:{person_id}` | 24 h |
| LP dedup | Redis Set | `lp:{plate_str}:{stream_id}` | 30 s |
| Track last-seen | Redis Hash | `track:lastseen:{track_id}` | 10 min |

---

## Operational Notes

**Probe budget**: Target < 1 ms per frame in the probe. The probe does zero I/O and zero encoding — only metadata copy, quality score computation (arithmetic), and queue push.

**Queue sizing**: Set `crop_queue maxsize` to ~2× your peak frame-rate × worker latency product. Drop oldest on full rather than blocking.

**JPEG encode offload**: `extract_and_encode_crop` (GPU→CPU + turbojpeg encode) typically takes 2–5 ms. Run in a thread pool (`asyncio.to_thread`) so it never touches the pipeline thread.

**Crop size policy**: Inline base64 for crops < 10 KB (LP regions). Use S3/MinIO + key for person/vehicle crops to keep Kafka message sizes bounded.

**Track ID stability**: nvtracker resets IDs on pipeline restart. Always include `pipeline_instance_id` (UUID generated at startup) in every message so consumers can detect restarts and flush stale state.

**Kafka producer config (librdkafka)**:
```ini
queue.buffering.max.messages=100000
queue.buffering.max.ms=5
batch.num.messages=500
compression.type=lz4
message.max.bytes=5242880   # 5 MB — covers base64 person crop
```

**Consumer scaling**:
- ReID Consumer: scale by camera zone, not by camera count (gallery is per-zone)
- LP OCR Consumer: stateless, scale horizontally by partition count
- Track Aggregator: one instance per `stream_id` partition to avoid cross-instance state
- Analytics Consumer: stateless, scale freely
