# ⚽ Soccer Video Understanding System

An end-to-end computer-vision pipeline that detects players, goalkeepers, referees, and the ball in soccer footage, builds a second-by-second timeline of the match, and lets you ask questions about the video in plain English — using a language model running entirely on your own machine.

Built and trained from scratch on a single NVIDIA GPU.

## What it does

- **Detects** players, goalkeepers, referees, and the ball using a custom-trained YOLO model.
- **Builds a timeline** recording what was on screen each second.
- **Answers questions** about the match (e.g. *"How many players were visible?"*, *"When did the referee appear?"*) via a local LLM — no cloud, no API keys.

## Pipeline

`video → object detection (custom YOLO model) → per-second timeline → LLM question-answering`

## Tech stack

- **Python**, **PyTorch**, **CUDA** (GPU acceleration)
- **Ultralytics YOLO26** — object detection
- **OpenCV** — video processing
- **Ollama** + **Llama 3.2** — local LLM for Q&A
- Dataset: [Roboflow football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)

## Results

The detector was fine-tuned on a 372-image soccer dataset (4 classes) and trained locally on an NVIDIA GTX 1650. Inference runs at ~20 ms/frame (real-time) on the GPU.

| Class       | mAP@50 |
|-------------|--------|
| Player      | 0.95   |
| Goalkeeper  | 0.79   |
| Referee     | 0.70   |
| Ball        | 0.22   |
| **Overall** | **0.66** |

## Scripts

| File | Purpose |
|------|---------|
| `train.py` | Fine-tune the soccer model on the dataset |
| `detect_soccer.py` | Run the trained model on a video |
| `timeline.py` | Build a per-second detection timeline (`timeline.json`) |
| `qa.py` | Ask natural-language questions about the timeline |

## Limitations & lessons learned

- **Domain shift:** performs best on broadcast-style match footage (like its training data); less reliable on amateur/phone footage from unusual angles.
- **Ball detection is hard:** the ball is small, fast, and motion-blurred — the lowest-accuracy class, a known challenge even in professional sports analytics.
- **Per-player events** (passes, shots, ratings) are intentionally out of scope; they require player re-identification, possession tracking, and event detection — research-grade work.

## Future work

- Team classification via jersey-color clustering
- Ball-possession estimation and player tracking
- Computing match statistics in code (rather than relying on the LLM for arithmetic)
