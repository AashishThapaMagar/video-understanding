from ultralytics import YOLO
import cv2
import json
from collections import defaultdict

# Load your trained soccer model
model = YOLO(r"C:\Users\Aashish\runs\detect\train-6\weights\best.pt")

video_path = "soccer.mp4"

# Find the video's frames-per-second, so we can turn frame numbers into seconds
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()

# Run detection frame by frame (stream=True is memory-friendly for video)
results = model.predict(video_path, stream=True, conf=0.1, verbose=False)

# For each second, record the most of each class we saw
per_second = defaultdict(lambda: defaultdict(int))

for frame_index, r in enumerate(results):
    second = int(frame_index / fps)
    counts = defaultdict(int)
    for box in r.boxes:
        label = model.names[int(box.cls[0])]
        counts[label] += 1
    for label, n in counts.items():
        per_second[second][label] = max(per_second[second][label], n)

# Build a clean, ordered timeline
timeline = []
for second in sorted(per_second):
    entry = {"second": second, **per_second[second]}
    timeline.append(entry)

# Save it to a file for the Q&A step later
with open("timeline.json", "w") as f:
    json.dump(timeline, f, indent=2)

# Print a readable version
print("\n--- Match timeline ---")
for entry in timeline:
    parts = [f"{v} {k}" for k, v in entry.items() if k != "second"]
    summary = ", ".join(parts) if parts else "nothing detected"
    print(f"{entry['second']:02d}s: {summary}")

print(f"\nSaved timeline to timeline.json ({len(timeline)} seconds)")