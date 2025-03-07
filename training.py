import torch
import datetime
import os
from pathlib import Path

if __name__ == '__main__':
    try:
        from ultralytics import YOLO

        model_path = "./weights/yolo_nano_solar_panel.pt"
        model = YOLO(model_path)
        
        # Create output directory if it doesn't exist
        output_dir = Path("./runs")
        output_dir.mkdir(exist_ok=True)
        
        # Train the model with additional parameters for better control
        train_results = model.train(
            data="./dataset/data.yaml",  # path to dataset YAML
            epochs=100,                  # number of training epochs
            imgsz=640,                   # training image size
            device=0,                    # device to run on (converts string "0" to integer 0)
            batch=16,                    # batch size
            save=True,                   # save checkpoints
            project=str(output_dir),     # project directory
            name="train",                # experiment name
            exist_ok=True,               # overwrite existing experiment
            patience=50,                 # early stopping patience
            pretrained=True,             # use pretrained model
        )
        
        # Evaluate model performance on the validation set
        val_results = model.val()
        print(f"Validation results: {val_results}")
        
        # Export the model to ONNX format with explicit input/output names
        try:
            export_path = model.export(
                format="onnx",
                opset=12,               # ONNX opset version
                half=True,              # FP16 quantization
                simplify=True,          # Simplify ONNX model
            )
            print(f"Model exported to: {export_path}")
        except Exception as e:
            print(f"Export error: {e}")
        
        # Get date and time in format %DD%MM%YYYY_%HH%MM%SS
        now = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # Save the model with structured naming
        save_path = output_dir / f"yolov8n_{now}.pt"
        model.save(str(save_path))
        print(f"Model saved to: {save_path}")
        
        # Save additional formats if needed
        try:
            ts_path = output_dir / f"yolov8n_ts_{now}.pt"
            model.export(format="torchscript", path=str(ts_path))
            print(f"TorchScript model saved to: {ts_path}")
        except Exception as e:
            print(f"TorchScript export error: {e}")
            
    except ImportError as e:
        print(f"Error: {e}. Please install required packages with 'pip install ultralytics'")
    except Exception as e:
        print(f"An error occurred: {e}")