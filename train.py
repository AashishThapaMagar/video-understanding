from ultralytics import YOLO
from pathlib import Path
import yaml


def main():
    # --- Fix the dataset paths so YOLO can find the images ---
    data_dir = Path("soccer-data").resolve()
    yaml_path = data_dir / "data.yaml"

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(data_dir)
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Train a soccer detector from pretrained YOLO26 nano ---
    model = YOLO("yolo26n.pt")
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=4,      # small batch for your 4GB GPU
        device=0,     # 0 = run on your GPU
        workers=0,
    )


if __name__ == "__main__":
    main()