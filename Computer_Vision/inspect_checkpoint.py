import torch
import pathlib
import os
import sys

# --- Setup paths ---
# Assuming this script is in D:\Downloads\Computer_Vision\Computer_Vision
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(PROJECT_ROOT, 'models', 'best.pt')
YOLOV5_REPO_PATH = os.path.join(PROJECT_ROOT, 'yolov5')

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"MODEL_FILE_PATH: {MODEL_FILE_PATH}")
print(f"YOLOV5_REPO_PATH: {YOLOV5_REPO_PATH}")

if not os.path.exists(MODEL_FILE_PATH):
    print(f"ERROR: Model file not found: {MODEL_FILE_PATH}")
    sys.exit(1)
if not os.path.isdir(YOLOV5_REPO_PATH):
    print(f"ERROR: YOLOv5 repo not found: {YOLOV5_REPO_PATH}")
    sys.exit(1)

# --- Environment manipulation (similar to PPEDetector) ---
original_cwd = os.getcwd()
original_sys_path = list(sys.path)

print(f"\nOriginal CWD: {original_cwd}")
# print(f"Original sys.path: {original_sys_path}")

try:
    print(f"Changing CWD to: {YOLOV5_REPO_PATH}")
    os.chdir(YOLOV5_REPO_PATH)

    # Add YOLOv5 repo to sys.path so 'models.yolo' can be found
    # This makes 'models' (inside yolov5_repo_path) a top-level package for the import system
    if YOLOV5_REPO_PATH not in sys.path:
        sys.path.insert(0, YOLOV5_REPO_PATH)
    print(f"Modified sys.path[0]: {sys.path[0]}")

    print(f"\nAttempting to load checkpoint: {MODEL_FILE_PATH}")
    # When loading a model saved as an entire object, it's often better to let PyTorch
    # find the classes from the modified sys.path.
    # Using weights_only=False is necessary if the .pt file contains more than just weights.
    ckpt = torch.load(MODEL_FILE_PATH, map_location='cpu') # weights_only defaults to False
    print(f"Checkpoint loaded successfully. Type: {type(ckpt)}")

    # --- (Optional) Inspection code from previous version ---
    if isinstance(ckpt, dict):
        print("\nCheckpoint is a dictionary. Inspecting keys and value types:")
        for key, value in ckpt.items():
            value_type_str = str(type(value))
            print(f"  Key: '{key}', Value Type: {value_type_str}")
            if "PosixPath" in value_type_str: # Check for PosixPath again
                print(f"    !!!! Found PosixPath object under key '{key}' !!!! Value: {value}")
            # ... (other inspection logic if needed) ...
    elif hasattr(ckpt, 'state_dict'): # It's a model object
        print("\nCheckpoint is a model object. Inspecting its attributes:")
        for attr_name, attr_value in vars(ckpt).items():
            attr_type_str = str(type(attr_value))
            if "PosixPath" in attr_type_str: # Check for PosixPath again
                print(f"    !!!! Found PosixPath attribute: '{attr_name}' !!!! Value: {attr_value}")
            # ... (other inspection logic if needed) ...
    else:
        print("\nCheckpoint is not a dictionary or a model object with a state_dict.")


except Exception as e:
    print(f"\nAn error occurred during torch.load: {e}")
    import traceback
    traceback.print_exc() # Print full traceback

finally:
    print("\nRestoring original environment...")
    os.chdir(original_cwd)
    sys.path = original_sys_path
    print(f"Restored CWD: {os.getcwd()}")
    # print(f"Restored sys.path: {sys.path}")
