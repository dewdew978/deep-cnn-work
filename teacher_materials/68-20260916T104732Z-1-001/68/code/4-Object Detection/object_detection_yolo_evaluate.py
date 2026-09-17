from ultralytics import YOLO

def validate_model():
    # Load a model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Customize validation settings
    validation_results = model.val(data="mock_data_file.yaml", imgsz=256)


if __name__ == "__main__":
    validate_model()