import cv2
import os
from detection.ppe_detector import PPEDetector
from tracking.object_tracker import ObjectTracker 
from association.ppe_associator import PPEAssociator
from compliance_checker.safety_rules import SafetyComplianceChecker
from project_utils.video_utils import draw_tracked_ppe_status

def main(video_path, model_path, output_video_path=None):
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model path is correct and the model file exists.")
        return

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please ensure the video path is correct and the video file exists.")
        return

    detector = PPEDetector(model_path=model_path, confidence_threshold=0.4) 
    if not detector.model: 
        print("Failed to load the model. Exiting.")
        return

    tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    # Parameters for PPEAssociator might need tuning
    associator = PPEAssociator(iou_threshold_person_ppe=0.05, helmet_y_offset_factor=0.15, vest_overlap_factor=0.3)
    compliance_checker = SafetyComplianceChecker(require_helmet=True, require_vest=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    writer = None
    if output_video_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None: 
            print("Warning: Video FPS is 0 or None. Defaulting to 25 FPS for writer.")
            fps = 25 
        
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"Error creating output directory {output_dir}: {e}")
                # Decide if to return or continue without writing
                output_video_path = None 

        if output_video_path: # Re-check in case it was disabled
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            if not writer.isOpened():
                print(f"Error: Could not open video writer for path {output_video_path}")
                writer = None # Ensure writer is none if opening failed


    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("End of video or cannot read frame.")
            break
        
        frame_idx += 1
        print(f"Processing frame {frame_idx}...")

        # Detection: returns list of [x1, y1, x2, y2, confidence, class_id, class_name]
        all_detections = detector.detect(frame.copy()) 
        
        if not all_detections:
            # If no detections, still write the original frame to the output video
            if writer: 
                writer.write(frame)
            # cv2.imshow('PPE Compliance Monitoring', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break 
            continue

        # Tracking: update expects detections in a specific format.
        # ObjectTracker's update should return: [[x1,y1,x2,y2,track_id,cls_id,name], ...]
        all_tracked_objects = tracker.update(all_detections)

        # Filter for tracked persons
        # Ensure obj[6] (class_name) exists and is correct
        tracked_persons = [obj for obj in all_tracked_objects if len(obj) > 6 and obj[6] == 'person']
        
        # Association
        # associate_ppe_to_persons expects tracked_persons and all_tracked_objects
        person_ppe_associations = associator.associate_ppe_to_persons(tracked_persons, all_tracked_objects)
        
        # Compliance Checking
        ppe_violations = compliance_checker.check_ppe_compliance(person_ppe_associations)

        # Visualization
        # draw_tracked_ppe_status needs tracked_persons, associations, violations, AND all_tracked_objects
        output_frame = draw_tracked_ppe_status(frame.copy(), tracked_persons, person_ppe_associations, ppe_violations, all_tracked_objects)
        
        if writer:
            writer.write(output_frame)
        # cv2.imshow('PPE Compliance Monitoring', output_frame)
        
        # key = cv2.waitKey(1) & 0xFF # Commented for Colab
        # if key == ord('q'):
        #     print("Quitting...")
        #     break
        # elif key == ord('p'): 
        #     print("Paused. Press any key to continue...")
        #     cv2.waitKey(-1) # cv2.waitKey(0) also works for indefinite pause


    cap.release()
    if writer: 
        writer.release()
        print(f"Output video saved to {output_video_path}")
    # cv2.destroyAllWindows() # Commented for Colab
    print("Processing finished.")

if __name__ == '__main__':
    # --- Define paths for Colab ---
    # Assuming your project 'Computer_Vision' is cloned into /content/
    # and your script main_app.py is in /content/Computer_Vision/Computer_Vision/src/
    
    # Get the directory of the current script (src)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to Computer_Vision/Computer_Vision/ (which is project_base_dir)
    project_base_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

    VIDEO_PATH = os.path.join(project_base_dir, 'sample_videos', 'video_test.mp4')
    MODEL_WEIGHTS_PATH = os.path.join(project_base_dir, 'models', 'best.pt')
    OUTPUT_VIDEO = os.path.join(project_base_dir, 'output_videos', 'output_ppe_compliance_colab.mp4')

    # Ensure output directory for video exists
    output_video_dir = os.path.dirname(OUTPUT_VIDEO)
    if not os.path.exists(output_video_dir):
        try:
            os.makedirs(output_video_dir)
            print(f"Created output video directory: {output_video_dir}")
        except OSError as e:
            print(f"Error creating output video directory {output_video_dir}: {e}")
            # Decide if you want to exit or try to save in current dir
            OUTPUT_VIDEO = os.path.basename(OUTPUT_VIDEO) # Fallback to current dir
            print(f"Warning: Will attempt to save output to current directory as: {OUTPUT_VIDEO}")


    print(f"Attempting to process video: {VIDEO_PATH}")
    print(f"Using model: {MODEL_WEIGHTS_PATH}")
    print(f"Output will be saved to: {OUTPUT_VIDEO}")
    
    main(VIDEO_PATH, MODEL_WEIGHTS_PATH, OUTPUT_VIDEO)