import cv2
import numpy as np

# Define some colors (BGR format) - Consider making these brighter if needed
COLOR_GREEN = (0, 255, 0)       # Compliant / Default Person
COLOR_RED = (0, 0, 255)         # Violation Person
COLOR_BLUE = (255, 100, 100)    # Helmet (slightly lighter blue)
COLOR_YELLOW = (50, 255, 255)   # Vest (brighter yellow)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0,0,0)
COLOR_CYAN = (255, 255, 50)     # Default for other tracked objects (brighter cyan)
COLOR_MAGENTA = (255, 50, 255)  # (brighter magenta)
COLOR_ORANGE = (0, 165, 255)    # Example of another bright color

# --- Adjustable Drawing Parameters ---
BOX_THICKNESS = 3 # Increased from 2
LABEL_FONT_SCALE = 0.6 # Increased from 0.5
LABEL_FONT_THICKNESS = 2 # Increased from 1
LABEL_TEXT_COLOR_ON_BG = COLOR_BLACK # Text color on the label background
LABEL_BG_OPACITY = 0.7 # Opacity of the label background (0.0 to 1.0)
# --- End of Adjustable Parameters ---


def draw_tracked_ppe_status(frame, tracked_persons, person_ppe_associations, ppe_violations, all_tracked_objects):
    """
    Draws bounding boxes for tracked persons, their associated PPE, violation status,
    and all other tracked objects with enhanced visibility.
    """
    drawn_ppe_track_ids = set() 

    # 1. Draw Persons and their Associated PPE
    for person_data in tracked_persons:
        if len(person_data) < 5: 
            continue
        try:
            px1, py1, px2, py2 = map(int, person_data[0:4])
            person_id = int(person_data[4])
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse coordinates/ID for person: {person_data}. Error: {e}")
            continue
            
        person_box_color = COLOR_GREEN 
        status_lines = [f"Person {person_id}"]
        # Determine compliance status color for text (not the box itself initially)
        compliance_text_color = COLOR_GREEN 

        if person_id in ppe_violations:
            violation_details = ppe_violations[person_id]
            person_box_color = COLOR_RED 
            compliance_text_color = COLOR_RED
            if violation_details.get('missing_helmet'):
                status_lines.append("MISSING HELMET")
            if violation_details.get('missing_vest'):
                status_lines.append("MISSING VEST")
            if len(status_lines) == 1: 
                status_lines.append("NON-COMPLIANT")
        
        cv2.rectangle(frame, (px1, py1), (px2, py2), person_box_color, BOX_THICKNESS)

        if person_id in person_ppe_associations:
            associated_ppe = person_ppe_associations[person_id]
            helmet_info = associated_ppe.get('helmet_bbox')
            if helmet_info and len(helmet_info) >= 4:
                try:
                    hx1, hy1, hx2, hy2 = map(int, helmet_info[0:4])
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), COLOR_BLUE, BOX_THICKNESS -1 if BOX_THICKNESS > 1 else 1) # Slightly thinner for associated PPE
                    if len(helmet_info) >= 5: 
                        drawn_ppe_track_ids.add(int(helmet_info[4]))
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Warning: Could not parse/draw helmet for person {person_id}: {helmet_info}. Error: {e}")

            vest_info = associated_ppe.get('vest_bbox')
            if vest_info and len(vest_info) >= 4:
                try:
                    vx1, vy1, vx2, vy2 = map(int, vest_info[0:4])
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), COLOR_YELLOW, BOX_THICKNESS -1 if BOX_THICKNESS > 1 else 1) # Slightly thinner
                    if len(vest_info) >= 5: 
                        drawn_ppe_track_ids.add(int(vest_info[4]))
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Warning: Could not parse/draw vest for person {person_id}: {vest_info}. Error: {e}")

        # Draw status text for the person
        text_y_start_point = py1 - 7 # Starting y point for the first line of text (above the box)
        line_height = int(LABEL_FONT_SCALE * 25) + 5 # Approximate height of a line of text + padding
        
        # Calculate background size
        max_text_width = 0
        for line in status_lines:
            (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
            if text_w > max_text_width:
                max_text_width = text_w
        
        bg_height = len(status_lines) * line_height
        bg_pt1_y = text_y_start_point - bg_height + (line_height // 3) # Adjust to align better
        bg_pt1_x = px1
        
        # Ensure background is not drawn off-screen (top)
        if bg_pt1_y < 0:
            # If text background goes off screen top, try drawing below the top of the person box
            bg_pt1_y = py1 + 5 
            text_y_start_point = bg_pt1_y + line_height - (line_height // 3)


        bg_pt1 = (bg_pt1_x, bg_pt1_y)
        bg_pt2 = (bg_pt1_x + max_text_width + 10, bg_pt1_y + bg_height) # +10 for x-padding

        # Draw semi-transparent background for text
        if bg_pt1[1] < bg_pt2[1] and bg_pt1[0] < bg_pt2[0]: # Check if rectangle is valid
            try:
                sub_frame_for_bg = frame[bg_pt1[1]:bg_pt2[1], bg_pt1[0]:bg_pt2[0]]
                bg_color_rect = np.full(sub_frame_for_bg.shape, (200, 200, 200), dtype=np.uint8) # Light gray background
                
                # Blend background
                blended_bg = cv2.addWeighted(sub_frame_for_bg, 1 - LABEL_BG_OPACITY, bg_color_rect, LABEL_BG_OPACITY, 0)
                frame[bg_pt1[1]:bg_pt2[1], bg_pt1[0]:bg_pt2[0]] = blended_bg
            except Exception as e:
                print(f"Warning: Could not draw text background for person {person_id}. Error: {e}, Coords: {bg_pt1}, {bg_pt2}")


        # Draw each line of text
        for i, line in enumerate(status_lines):
            text_y_pos = text_y_start_point - ( (len(status_lines) - 1 - i) * line_height )
            # If background was shifted down, adjust text y position accordingly
            if bg_pt1_y == py1 + 5: # check if background was shifted
                 text_y_pos = bg_pt1_y + (i * line_height) + (line_height //3)*2


            current_line_color = compliance_text_color if i > 0 else COLOR_WHITE # "Person ID" in white, status lines in compliance color
            
            # Draw outline (optional, can make text pop more)
            # cv2.putText(frame, line, (bg_pt1_x + 5, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, COLOR_BLACK, LABEL_FONT_THICKNESS + 1)
            cv2.putText(frame, line, (bg_pt1_x + 5, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, current_line_color, LABEL_FONT_THICKNESS)


    # 2. Draw All Other Tracked Objects
    for obj_data in all_tracked_objects:
        if len(obj_data) < 7: 
            continue
        try:
            obj_x1, obj_y1, obj_x2, obj_y2 = map(int, obj_data[0:4])
            obj_track_id = int(obj_data[4])
            obj_class_name = obj_data[6]
        except (ValueError, TypeError, IndexError) as e:
            print(f"Warning: Could not parse data for a generic tracked object: {obj_data}. Error: {e}")
            continue

        if obj_class_name == 'person' or obj_track_id in drawn_ppe_track_ids:
            continue

        obj_color = COLOR_CYAN 
        if obj_class_name == 'helmet':
            obj_color = COLOR_BLUE
        elif obj_class_name == 'vest':
            obj_color = COLOR_YELLOW
        elif 'no-helmet' in obj_class_name: 
            obj_color = COLOR_ORANGE # Use a different distinct color
        elif 'no-vest' in obj_class_name:
            obj_color = COLOR_ORANGE
        
        cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), obj_color, BOX_THICKNESS -1 if BOX_THICKNESS > 1 else 1) # Slightly thinner for other objects
        
        label = f"{obj_class_name} {obj_track_id}"
        label_y_pos = obj_y1 - 7
        if label_y_pos < 10 : label_y_pos = obj_y1 + int(LABEL_FONT_SCALE * 25) # Draw below if too close to top

        # Simple background for these labels too
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)
        label_bg_pt1 = (obj_x1, label_y_pos - label_h - 3)
        label_bg_pt2 = (obj_x1 + label_w + 4, label_y_pos + 3)
        
        if label_bg_pt1[1] < 0 : # Adjust if going off screen top
            label_bg_pt1 = (obj_x1, obj_y1 + 2)
            label_bg_pt2 = (obj_x1 + label_w + 4, obj_y1 + label_h + 7)
            label_y_pos = obj_y1 + label_h + 2


        try:
            if label_bg_pt1[1] < label_bg_pt2[1] and label_bg_pt1[0] < label_bg_pt2[0]:
                sub_frame_for_label_bg = frame[label_bg_pt1[1]:label_bg_pt2[1], label_bg_pt1[0]:label_bg_pt2[0]]
                label_bg_color_rect = np.full(sub_frame_for_label_bg.shape, (180, 180, 180), dtype=np.uint8) # Slightly darker gray
                blended_label_bg = cv2.addWeighted(sub_frame_for_label_bg, 1 - LABEL_BG_OPACITY, label_bg_color_rect, LABEL_BG_OPACITY, 0)
                frame[label_bg_pt1[1]:label_bg_pt2[1], label_bg_pt1[0]:label_bg_pt2[0]] = blended_label_bg
        except Exception as e:
             print(f"Warning: Could not draw text background for object {obj_track_id}. Error: {e}")


        cv2.putText(frame, label, (obj_x1 + 2, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, obj_color, LABEL_FONT_THICKNESS)

    return frame

if __name__ == '__main__':
    dummy_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    dummy_frame[:] = (100, 100, 100) 

    mock_tracked_persons = [
        [50.0, 50.0, 150.0, 250.0, 1, 0, 'person'],
        [200.0, 300.0, 280.0, 550.0, 2, 0, 'person'] 
    ]
    mock_person_ppe_associations = {
        1: {'helmet_bbox': [70.0, 30.0, 130.0, 80.0, 101], 'vest_bbox': [60.0, 100.0, 140.0, 200.0, 102]},
        2: {'vest_bbox': [210.0, 350.0, 270.0, 450.0, 105]}
    }
    mock_ppe_violations = {
        2: {'missing_helmet': True}
    }
    
    mock_all_tracked_objects = [
        [50.0, 50.0, 150.0, 250.0, 1, 0, 'person'],      
        [70.0, 30.0, 130.0, 80.0, 101, 1, 'helmet'],    
        [60.0, 100.0, 140.0, 200.0, 102, 2, 'vest'],     
        [300.0, 50.0, 350.0, 100.0, 103, 1, 'helmet'],  
        [400.0, 100.0, 480.0, 180.0, 104, 2, 'vest'],   
        [200.0, 300.0, 280.0, 550.0, 2, 0, 'person'],
        [210.0, 350.0, 270.0, 450.0, 105, 2, 'vest'],
        [500.0, 70.0, 580.0, 120.0, 106, 3, 'no-helmet']
    ]

    output_frame = draw_tracked_ppe_status(dummy_frame.copy(), 
                                           mock_tracked_persons, 
                                           mock_person_ppe_associations, 
                                           mock_ppe_violations,
                                           mock_all_tracked_objects)

    cv2.imwrite("ppe_status_enhanced_visibility_demo.jpg", output_frame)
    print("Saved demo image to ppe_status_enhanced_visibility_demo.jpg")