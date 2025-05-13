import numpy as np

class PPEAssociator:
    def __init__(self, iou_threshold_person_ppe=0.1, helmet_y_offset_factor=0.1, vest_overlap_factor=0.4):
        self.iou_threshold_person_ppe = iou_threshold_person_ppe
        self.helmet_y_offset_factor = helmet_y_offset_factor # For helmet relative position to person's head
        self.vest_overlap_factor = vest_overlap_factor # For vest relative position to person's torso

    def _calculate_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        denominator = float(boxAArea + boxBArea - interArea)
        if denominator == 0:
            return 0
        
        iou = interArea / (denominator + 1e-6) # Add epsilon to avoid division by zero
        return iou

    def associate_ppe_to_persons(self, tracked_persons, all_tracked_objects):
        """
        Associate PPE (helmet, vest, no-helmet, no-vest) with each tracked person.
        :param tracked_persons: List of tracked persons [[x1,y1,x2,y2,id,cls_id,name], ...]
        :param all_tracked_objects: List of ALL tracked objects (including PPE).
                                     [[x1,y1,x2,y2,id,cls_id,name], ...]
        :return: Dictionary, key is person_track_id,
                 value is {'helmet_status': 'helmet'/'no-helmet'/'unknown',
                           'vest_status': 'vest'/'no-vest'/'unknown',
                           'bbox': person_box}
        """
        person_ppe_status = {}

        # Separate PPE objects based on class_name (index 6)
        tracked_helmets = [obj for obj in all_tracked_objects if obj[6] == 'helmet']
        tracked_no_helmets = [obj for obj in all_tracked_objects if obj[6] == 'no-helmet']
        tracked_vests = [obj for obj in all_tracked_objects if obj[6] == 'vest']
        tracked_no_vests = [obj for obj in all_tracked_objects if obj[6] == 'no-vest']

        for p_idx, person_track in enumerate(tracked_persons):
            px1, py1, px2, py2, person_id, _, _ = person_track[:7] # Ensure we only unpack expected values
            person_box = (px1, py1, px2, py2)
            person_height = py2 - py1
            person_width = px2 - px1
            
            person_ppe_status[person_id] = {
                'helmet_status': 'unknown', # Default is unknown
                'vest_status': 'unknown',   # Default is unknown
                'bbox': person_box
            }

            # 1. Check for 'no-helmet' detections associated with the person
            has_no_helmet_detection = False
            for no_helmet_track in tracked_no_helmets:
                nhx1, nhy1, nhx2, nhy2, _, _, _ = no_helmet_track[:7]
                no_helmet_box = (nhx1, nhy1, nhx2, nhy2)
                iou_person_no_helmet = self._calculate_iou(person_box, no_helmet_box)
                
                # 'no-helmet' is often a box around the head.
                # Check for significant overlap and relative position (upper part of person box)
                if iou_person_no_helmet > 0.3: # Requires decent IoU with the person
                    no_helmet_center_y = (nhy1 + nhy2) / 2
                    # Check if 'no-helmet' box is in the upper region of the person box
                    if py1 < no_helmet_center_y < (py1 + person_height * 0.35): # Top 35% of person height
                        person_ppe_status[person_id]['helmet_status'] = 'no-helmet'
                        has_no_helmet_detection = True
                        break
            
            # 2. If no 'no-helmet' detected, check for 'helmet'
            if not has_no_helmet_detection:
                best_helmet_iou = 0
                associated_helmet = False
                for helmet_track in tracked_helmets:
                    hx1, hy1, hx2, hy2, _, _, _ = helmet_track[:7]
                    helmet_box = (hx1, hy1, hx2, hy2)
                    iou_person_helmet = self._calculate_iou(person_box, helmet_box)

                    if iou_person_helmet > self.iou_threshold_person_ppe: # Basic IoU check
                        helmet_center_y = (hy1 + hy2) / 2
                        helmet_center_x = (hx1 + hx2) / 2
                        
                        # Check vertical position: helmet should be above or at the very top of the person
                        # Check horizontal position: helmet center should be within person's width
                        # The helmet_y_offset_factor helps define the expected vertical region for a helmet.
                        # A helmet should be roughly from slightly above the person's head (py1) to a bit down.
                        expected_helmet_y_top = py1 - person_height * self.helmet_y_offset_factor
                        expected_helmet_y_bottom = py1 + person_height * self.helmet_y_offset_factor * 2 # Allow some overlap into head
                        
                        if expected_helmet_y_top < helmet_center_y < expected_helmet_y_bottom and \
                           px1 < helmet_center_x < px2:
                            if iou_person_helmet > best_helmet_iou : # Prioritize higher IoU if multiple helmets overlap
                                best_helmet_iou = iou_person_helmet
                                person_ppe_status[person_id]['helmet_status'] = 'helmet'
                                associated_helmet = True
                if not associated_helmet and person_ppe_status[person_id]['helmet_status'] == 'unknown':
                    pass # Remains 'unknown' if no 'no-helmet' and no 'helmet' found

            # 3. Check for 'no-vest' detections
            has_no_vest_detection = False
            for no_vest_track in tracked_no_vests:
                nvx1, nvy1, nvx2, nvy2, _, _, _ = no_vest_track[:7]
                no_vest_box = (nvx1, nvy1, nvx2, nvy2)
                iou_person_no_vest = self._calculate_iou(person_box, no_vest_box)
                
                # 'no-vest' might be a box around the torso area.
                # Requires high IoU and should cover a significant portion of the torso.
                if iou_person_no_vest > 0.4: # Higher IoU threshold for 'no-vest'
                    no_vest_center_y = (nvy1 + nvy2) / 2
                    # Check if 'no-vest' box is in the torso region (e.g., middle 50% of person height)
                    if (py1 + person_height * 0.2) < no_vest_center_y < (py2 - person_height * 0.2):
                        person_ppe_status[person_id]['vest_status'] = 'no-vest'
                        has_no_vest_detection = True
                        break

            # 4. If no 'no-vest' detected, check for 'vest'
            if not has_no_vest_detection:
                best_vest_iou = 0
                associated_vest = False
                for vest_track in tracked_vests:
                    vx1, vy1, vx2, vy2, _, _, _ = vest_track[:7]
                    vest_box = (vx1, vy1, vx2, vy2)
                    iou_person_vest = self._calculate_iou(person_box, vest_box)
                    
                    if iou_person_vest > self.vest_overlap_factor: # Use specified overlap factor
                        vest_center_x = (vx1 + vx2) / 2
                        vest_center_y = (vy1 + vy2) / 2
                        
                        # Vest should be centered on the person's torso
                        # Check if vest center is within person's bounding box (crude torso check)
                        if px1 < vest_center_x < px2 and (py1 + person_height * 0.1) < vest_center_y < (py2 - person_height * 0.1):
                             if iou_person_vest > best_vest_iou:
                                best_vest_iou = iou_person_vest
                                person_ppe_status[person_id]['vest_status'] = 'vest'
                                associated_vest = True
                if not associated_vest and person_ppe_status[person_id]['vest_status'] == 'unknown':
                    pass # Remains 'unknown'

        return person_ppe_status
