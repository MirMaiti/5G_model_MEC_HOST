# SignBridge

Sign recognition split across two machines: the **capture host** extracts hand
landmarks and sends those over TCP, a **MEC server** runs the model and sends a
prediction back, and the host speaks it.

```
   capture host                          TCP                MEC server
   ------------                     ------------            ----------
   camera
     |
   MediaPipe  ---->  42 landmarks  --- 568 B/frame --->  normalise -> features
                                                              |
                                                            model
                                                              |
   speech  <----  "hello"          <--- JSON reply ----   vote + threshold
```

The image never leaves the host. That is the whole point of the design, and it
buys three things:

- **Bandwidth.** One frame of hand landmarks is 568 bytes on the wire. The same
  frame as a 640×480 JPEG is 20–40 kB, so roughly **50× less traffic** — about
  17 kB/s at 30 fps.
- **A smaller server.** The MEC node never imports MediaPipe or OpenCV, and
  never decodes an image. It needs PyTorch and nothing else.
- **Privacy.** No video is transmitted, and the collector writes only
  coordinates to disk — never frames of you signing.

> **Nothing here is trained yet.** `models/` is empty by design. The pipeline is
> built and tested end to end, but the model is yours to train on your own
> recordings — see [Record your data](#3-record-your-data) and
> [Train](#4-train).

---

## Contents

1. [Install](#1-install)
2. [Check the link before you record anything](#2-check-the-link-before-you-record-anything)
3. [Record your data](#3-record-your-data)
4. [Train](#4-train)
5. [Run it](#5-run-it)
6. [Data format](#6-data-format)
7. [The wire protocol](#7-the-wire-protocol)
8. [Configuration](#8-configuration)
9. [Project layout](#9-project-layout)
10. [Extending it](#10-extending-it)
11. [Tests](#11-tests)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Install

The two ends have different dependencies, and neither needs the other's. Install
only what the machine actually does.

**Capture host** (camera, MediaPipe, speech — no PyTorch):

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements-host.txt
```

**MEC server** (PyTorch — no camera stack):

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements-server.txt
```

**One machine doing both**, plus the test suite:

```bash
pip install -r requirements-dev.txt
```

The host needs Google's MediaPipe landmarker bundle. MediaPipe 1.0 removed the
old `mp.solutions` API, so this is required, not optional:

```bash
python -m signbridge.cli.fetch_models
```

That downloads `hand_landmarker.task` into `models/`. If you already have a copy,
point at it instead: `python -m signbridge.cli.fetch_models --from-dir /path/to/models`.

---

## 2. Check the link before you record anything

Before collecting data, prove the network path works. `--smoke-test` serves an
**untrained** model — its predictions are meaningless, and it says so in the
handshake so the host refuses to speak them.

Terminal 1:

```bash
python -m signbridge.cli.serve --smoke-test
```

Terminal 2:

```bash
python -m signbridge.cli.run_host --source synthetic --count 150
```

You should see frames sent, predictions returned, and a round-trip figure. On
loopback that is single-digit milliseconds. Once this works, the transport is
not what is wrong with anything later.

---

## 3. Record your data

```bash
python -m signbridge.cli.collect --label hello   --samples 30
python -m signbridge.cli.collect --label thanks  --samples 30
python -m signbridge.cli.collect --label yes     --samples 30
python -m signbridge.cli.collect --label no      --samples 30
```

`SPACE` starts a clip, `q` quits. Each clip is 45 frames (~1.5 s) of landmarks
written to `data/train/<label>/`. Clips where hands were visible in fewer than
half the frames are discarded automatically.

What actually makes the model work:

- **At least 20–30 clips per sign.** Fewer and it memorises rather than learns.
- **Vary something between clips** — distance, angle, lighting, speed, which
  part of the frame you occupy. Thirty identical clips teach it your room.
- **Record the signs you will actually use.** Two signs is the minimum.

For a separate validation set instead of an automatic split, record into
`--split val`.

**Recruiting other people to record clips?** They don't need this whole repo,
a venv, or `signbridge` installed — send them
[`contribute_data.py`](contribute_data.py) on its own:

```bash
pip install mediapipe opencv-python numpy
python contribute_data.py --label hello --samples 30
```

It writes `.npz` clips in the exact same format as `signbridge.cli.collect`
(it auto-downloads the hand-landmarker model on first run), so the resulting
`data/train/<label>/` folder can be zipped up and dropped straight into this
project's `data/train/`, or pushed directly if they have write access to a
fork. Appends to a label that already exists rather than replacing it.

---

## 4. Train

```bash
python -m signbridge.cli.train --config config.yaml
```

This reads `data/train/`, holds out a stratified validation split (every sign
appears on both sides), trains the architecture from `config.yaml`, and writes
`models/best.pt` plus a `metrics.json` report with a confusion matrix.

The checkpoint carries the label list **and the exact feature description**, so
the server rebuilds the identical preprocessing. Change the feature settings
after training and loading fails loudly rather than feeding the model inputs it
never saw.

Useful overrides:

```bash
python -m signbridge.cli.train --architecture mlp --epochs 100 --device cpu
```

| Architecture | When |
|---|---|
| `mlp` | Static handshapes, where the pose alone identifies the sign. Fastest. |
| `gru` | Default. Signs that are a movement — order matters. |
| `transformer` | More capacity for a larger vocabulary; wants more data. |

---

## 5. Run it

On the MEC server:

```bash
python -m signbridge.cli.serve --config config.yaml
```

On the capture host:

```bash
python -m signbridge.cli.run_host --server 192.168.1.50:9009
```

A preview window shows the landmarks and the current prediction. Predictions are
spoken when a label holds steady for `stability_seconds` and has not been said
within `repeat_cooldown` — without that gate the host stutters the same word
several times a second.

Replay recorded clips instead of using a camera, which is how to test the server
and the speech output without a signer:

```bash
python -m signbridge.cli.run_host --source replay --clips data/train/hello --fps 30
```

---

## 6. Data format

Clips are `.npz` files. To bring your own data, write this and everything else
works unchanged:

| Key | Shape | Meaning |
|---|---|---|
| `landmarks` | `(T, 42, 3)` float32 | MediaPipe normalised x, y, z |
| `mask` | `(T, 42)` float32 | 1 = detected, 0 = absent. Optional but recommended |
| `label` | scalar string | The sign. Optional — see below |

Landmark order is fixed: indices `0–20` are the left hand, `21–41` the right, and
with `layout: hands_pose`, `42–74` the pose.

`mask` matters more than it looks: it is the only way to distinguish a landmark
genuinely measured at the origin from one that was never detected. Omit it and
it is inferred as "present unless exactly zero".

A clip's label is resolved in this order, so most existing datasets need no
reorganising:

1. the `label` field inside the `.npz`,
2. the directory name — `data/train/hello/clip_003.npz`,
3. the filename before the first `-` — `hello-a1b2c3.npz`.

Clips may be any length; they are resampled to the model's window at load.

---

## 7. The wire protocol

Raw TCP with a fixed 8-byte header. Landmarks are packed binary because they are
the bulk of the traffic; control messages and predictions are JSON because they
are rare and worth being able to read in a packet capture.

```
magic   2 bytes   b"SB"
version 1 byte
type    1 byte
length  4 bytes   uint32 big-endian
payload length bytes
```

A `FRAME` payload:

```
seq          4 bytes   uint32     echoed back, so the host can time the round trip
t_ms         8 bytes   float64    capture time
n_landmarks  2 bytes   uint16
coords       4·3L      float32    x, y, z per landmark
mask         L bytes   uint8
```

| Type | Direction | Payload |
|---|---|---|
| `HELLO` | host → server | Layout the host will send |
| `WELCOME` | server → host | Labels, window, feature dim, device |
| `FRAME` | host → server | One frame, packed binary |
| `PREDICTION` | server → host | Label, confidence, timings |
| `RESET` | host → server | Clear the rolling buffer |
| `ERROR` | server → host | A message was rejected |
| `PING`/`PONG` | either | Round-trip timing |
| `BYE` | either | Orderly shutdown |

Two properties worth knowing before writing another client:

- **It is not request/response.** The server buffers frames and answers every
  `inference_interval` frames. Keep sending; demultiplex replies by type.
- **`TCP_NODELAY` is set.** Nagle's algorithm would hold small landmark frames
  back waiting to coalesce, which is exactly wrong here.

---

## 8. Configuration

One `config.yaml`, read by every stage. An unknown key is an error, not a silent
no-op. The settings most worth tuning:

| Key | Effect |
|---|---|
| `layout` | `hands` (42) or `hands_pose` (75, for signs that depend on body position) |
| `model.window` | Frames per window. Keep equal to `data.clip_frames` |
| `server.inference_interval` | Lower = more responsive, more compute |
| `server.min_confidence` | Below this the server reports silence rather than guessing |
| `server.vote_window` | Majority vote over recent predictions; suppresses flicker |
| `tts.stability_seconds` | How long a label must hold before it is spoken |
| `tts.repeat_cooldown` | How long before the same phrase repeats |

---

## 9. Project layout

```
signbridge/
  landmarks.py       canonical landmark ordering, shared by every stage
  features.py        normalisation - the single source of truth
  sequence.py        resampling clips to a fixed window
  protocol.py        the TCP wire format
  config.py          the config tree

  capture/           host side: camera and landmark extraction
    base.py            LandmarkSource interface
    mediapipe_source.py
    replay_source.py   recorded clips and synthetic frames, for testing

  model/
    registry.py        name -> architecture
    architectures.py   mlp, gru, transformer
    checkpoint.py      weights + features + labels, saved together

  training/
    dataset.py         loading clips, label resolution, stratified split
    trainer.py         the training loop

  server/            MEC side
    predictor.py       the model seam
    session.py         per-connection buffer and smoothing
    tcp_server.py      the threaded TCP server

  host/
    client.py          the TCP client and session loop
    tts.py             speech backends and the gating rules

  cli/               collect, train, serve, run_host, fetch_models
```

---

## 10. Extending it

Each seam is a registry or an interface, so adding to one touches nothing else.

**A new architecture** — write it and decorate it; the name is then valid in
`config.yaml`:

```python
from signbridge.model.registry import register_architecture
from signbridge.model.architectures import SequenceClassifier

@register_architecture("tcn")
class TemporalConvNet(SequenceClassifier):
    def forward(self, x):        # (B, T, D) -> (B, C)
        ...
```

**A new speech backend** — subclass `TTSBackend` and register it:

```python
from signbridge.host.tts import TTSBackend, register_backend

@register_backend("cloud")
class CloudVoice(TTSBackend):
    def say(self, text: str) -> None:
        ...
```

**A new landmark source** — subclass `LandmarkSource` and yield `LandmarkFrame`s.
The client loop takes any source, so a video file, a second camera or a network
feed all work without touching the transport.

**A different inference backend** — implement `Predictor` (ONNX Runtime, a
quantised build, a remote accelerator). The TCP server only calls `.predict()`.

---

## 11. Tests

```bash
python -m pytest tests/ -q
```

82 tests, about 20 seconds. They cover the protocol (including messages split
across TCP segments, truncated payloads and foreign magic), the feature
invariances, buffering and vote smoothing, dataset loading, the speech gating
rules, and a full train → serve → predict round trip over a real socket.

`tests/test_dependency_isolation.py` asserts the architectural claim in a fresh
interpreter: the server imports no MediaPipe or OpenCV, and the host imports no
PyTorch.

---

## 12. Troubleshooting

**"Could not open camera 0"** — on macOS, camera permission belongs to the app
hosting your shell. An embedded terminal panel usually cannot get it; run from
Terminal.app.

**"MediaPipe model bundle not found"** — run `python -m signbridge.cli.fetch_models`.

**"This server serves the 'hands' layout; the host offered 'hands_pose'"** — the
two ends have different `layout` settings. They must match; the server's is
fixed by the checkpoint it loaded.

**"This checkpoint was built with feature version N"** — the feature maths
changed since the model was trained. Retrain it; serving it would feed the model
inputs it never saw.

**Predictions never rise above silence** — `server.min_confidence` may be too
high for a model trained on few clips, or the signs may be too similar. Check
the confusion matrix in `models/metrics.json`.

**It speaks too often, or repeats itself** — raise `tts.stability_seconds` and
`tts.repeat_cooldown`.

**Predictions lag behind the signing** — lower `server.inference_interval` and
`server.vote_window`. Both trade responsiveness against stability.
