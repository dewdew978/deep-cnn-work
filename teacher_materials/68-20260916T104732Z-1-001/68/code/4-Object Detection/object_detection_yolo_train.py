from ultralytics import YOLO
import yaml


def train_model():
    # Load the data.yaml file
    with open('mock_data_file.yaml', 'r') as file:
        data_config = yaml.safe_load(file)

    # Print the paths and check if they exist
    print(f"Training images: {data_config['train']}")
    print(f"Validation images: {data_config['val']}")
    print(f"Testing images: {data_config.get('test', 'Not specified')}")

    # Initialize the model
    model = YOLO("yolov8n.pt")

    # Train the model with your custom dataset
    results = model.train(data="mock_data_file.yaml", epochs=5, imgsz=256)

    print("Training completed")

if __name__ == "__main__":
    train_model()