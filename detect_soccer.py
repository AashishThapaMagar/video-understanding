from ultralytics import YOLO

# Load YOUR trained soccer model (adjust this path if your best.pt is elsewhere)
model = YOLO(r"C:\Users\Aashish\runs\detect\train-6\weights\best.pt")

# Run it on your soccer clip and save the annotated video
results = model.predict("soccer.mp4", save=True, conf=0.1)

print("\nDone! Your annotated video is in the runs folder.")