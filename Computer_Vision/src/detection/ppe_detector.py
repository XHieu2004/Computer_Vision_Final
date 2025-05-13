import torch
import os
import sys

class PPEDetector:
    def __init__(self, model_path, confidence_threshold=0.25):
        self.model_path = os.path.abspath(model_path) 
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        yolov5_repo_path = None
        original_cwd = os.getcwd()

        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..')) 
            yolov5_repo_path = os.path.join(project_root, 'yolov5')

            print(f"DEBUG: Resolved absolute model weights path: {self.model_path}")
            print(f"DEBUG: YOLOv5 repository path: {yolov5_repo_path}")

            if not os.path.isdir(yolov5_repo_path):
                raise FileNotFoundError(f"YOLOv5 repository not found at {yolov5_repo_path}.")
            if not os.path.exists(os.path.join(yolov5_repo_path, 'hubconf.py')):
                raise FileNotFoundError(f"hubconf.py not found in {yolov5_repo_path}.")

            # Change CWD to yolov5 directory for robust local loading
            os.chdir(yolov5_repo_path)
            print(f"DEBUG: Changed CWD to: {os.getcwd()}")
            
            print(f"Attempting to load custom model: {self.model_path}")
            self.model = torch.hub.load(
                repo_or_dir='.',  # Current directory (yolov5_repo_path)
                model='custom',
                path=self.model_path, # Absolute path to your .pt file
                source='local',
                force_reload=True, # Good to keep for avoiding hub cache issues
                trust_repo=True    
            )
            
            self.model.to(self.device)
            self.model.conf = self.confidence_threshold 
            print(f"YOLOv5 custom model loaded successfully using CWD '{yolov5_repo_path}'.")

        except Exception as e:
            print(f"Error loading model in PPEDetector: {e}")
            if yolov5_repo_path:
                 print(f"Attempted YOLOv5 repo path: {yolov5_repo_path}")
            print(f"Model weights path used: {self.model_path}")
            print(f"CWD during torch.hub.load attempt: {os.getcwd()}")
            self.model = None
        finally:
            os.chdir(original_cwd) # Restore original CWD
            print(f"DEBUG: Restored CWD to: {os.getcwd()}")

    def detect(self, frame):
        if self.model is None:
            print("Model not loaded, detection skipped.")
            return []
        results = self.model(frame) 
        detections = []
        if hasattr(results, 'xyxy') and results.xyxy[0].shape[0] > 0:
            for det in results.xyxy[0].cpu().numpy(): 
                x1, y1, x2, y2, conf, cls_id = det
                class_name = self.model.names[int(cls_id)] 
                detections.append([x1, y1, x2, y2, conf, int(cls_id), class_name])
        return detections

if __name__ == '__main__':
    example_project_root_from_detector = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    example_model_path_for_detector = os.path.join(example_project_root_from_detector, 'models', 'best.pt') 

    if not os.path.exists(example_model_path_for_detector):
        print(f"Test model file not found at {example_model_path_for_detector}.")
    else:
        print(f"Attempting to initialize PPEDetector with model: {example_model_path_for_detector}")
        detector = PPEDetector(model_path=example_model_path_for_detector)
        if detector.model:
            print("PPEDetector initialized successfully for testing.")
        else:
            print("PPEDetector initialization failed for testing.")