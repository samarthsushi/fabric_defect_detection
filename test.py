import torch
from pathlib import Path
import time
import os
from dotenv import load_dotenv

def load_model(model_path):
    """Load the YOLOv5 model from the given path."""
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def main():
    load_dotenv()
    model_path = os.getenv("YOLOV5_MODEL_PATH")
    save_path = "runs/detect/custom_test"

    try:
        model = load_model(model_path)
        print("Model loaded successfully.")

        while True:
            img_path = input("Enter path of image: ").strip().strip('"')
            if img_path.lower() == 'exit':
                break

            try:
                start_time = time.time()
                results = model(img_path)
                end_time = time.time()

                results.show()
                results.save(Path(save_path))
                print(f"Inference time: {end_time - start_time:.2f} seconds")

            except Exception as e:
                print(f"An error occurred while processing the image: {e}")

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

if __name__ == "__main__":
    main()
