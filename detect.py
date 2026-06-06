from ultralytics import YOLO

# Load YOLO26 nano — the small, fast model. It downloads itself the first time.
model = YOLO("yolo26n.pt")

# Run detection on a sample street image (people + a bus).
# save=True writes a copy with boxes drawn around everything it finds.
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True)

# Print, in plain English, what it detected.
for box in results[0].boxes:
    label = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"Found {label} — {confidence:.0%} confident")

print("\nDone. The annotated image is saved in the new 'runs' folder.")