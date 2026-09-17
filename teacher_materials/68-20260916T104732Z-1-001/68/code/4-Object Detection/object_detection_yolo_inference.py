import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def test_model():
    # Load the trained YOLO model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Define the path to the test images
    test_images_path = "CarDetectionYOLO/test/images"
    output_path = "prediction"  # Folder to save the predictions

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # List all the images in the test directory
    test_images = [f for f in os.listdir(test_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Perform inference on each test image
    for img_name in test_images:
        img_path = os.path.join(test_images_path, img_name)
        img = cv2.imread(img_path)
        
        # Run inference
        results = model(img)

        # Process results list
        for result in results:
            # Save the result image with predictions
            output_img_path = os.path.join(output_path, img_name)
            result.save(output_img_path)

    print("Inference completed. Results saved in the output directory.")

if __name__ == "__main__":
    test_model()
