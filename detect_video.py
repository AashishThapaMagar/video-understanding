from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Runs detection on your soccer clip and saves an annotated copy.
results = model.predict("soccer.mp4", save=True)

print("\nDone! The annotated video is in the 'runs' folder.")