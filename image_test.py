import torch
from ultralytics import YOLO

def main():
    # Load the YOLO11n model
    model = YOLO("./weights/yolo_nano_solar_panel.pt")

    # Evaluate model performance on the validation set
    print("Evaluating model performance...")
    metrics = model.val()
    print(metrics)

    # Perform object detection on a sample image
    image_path = "./sample_image.jpg"  # Replace with actual image path
    print(f"Running inference on: {image_path}")
    results = model(image_path)
    results[0].show()  # Display results


if __name__ == "__main__":
    main()
