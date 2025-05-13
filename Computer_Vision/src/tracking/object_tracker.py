import numpy as np
# Placeholder for a SORT-based tracker or similar.
# A full SORT implementation is complex. This is a simplified interface.
# The update method should take detections (e.g., [x1, y1, x2, y2, conf, cls_id])
# and return tracked objects (e.g., [x1, y1, x2, y2, track_id, cls_id, cls_name]).

class ObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_id_count = 0
        self.tracks = [] # Store active tracks: [x1,y1,x2,y2,id,cls_id,name,age,hits]

        # Note: A real SORT tracker would use Kalman Filters for state estimation
        # and the Hungarian algorithm for assignment. This is a conceptual placeholder.

    def _iou(self, bb_test, bb_gt):
        """
        Computes IoU between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6) # Add 1e-6 to avoid division by zero
        return o

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, conf, cls_id, cls_name]
        Returns: list of [x1, y1, x2, y2, track_id, cls_id, cls_name]
        """
        # This is a very simplified tracking logic for demonstration.
        # A proper SORT implementation is much more involved.
        
        # Predict new locations of tracks (skipped in this simplified version)

        # Associate detections with existing tracks
        matched_indices = []
        if len(self.tracks) > 0 and len(detections) > 0:
            track_bbs = np.array([track[:4] for track in self.tracks])
            det_bbs = np.array([det[:4] for det in detections])
            
            iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
            for d, det in enumerate(det_bbs):
                for t, trk in enumerate(track_bbs):
                    iou_matrix[d, t] = self._iou(det, trk)

            # Simple greedy matching (a real SORT uses Hungarian algorithm)
            for d_idx in range(len(detections)):
                best_match_t_idx = np.argmax(iou_matrix[d_idx, :])
                if iou_matrix[d_idx, best_match_t_idx] >= self.iou_threshold:
                    if best_match_t_idx not in [m[1] for m in matched_indices]:
                         # Update track with new detection
                        self.tracks[best_match_t_idx][:4] = detections[d_idx][:4] # Update bbox
                        self.tracks[best_match_t_idx][7] = 0 # Reset age
                        self.tracks[best_match_t_idx][8] = min(self.min_hits, self.tracks[best_match_t_idx][8] + 1) # Increment hits
                        # Keep original class_id and class_name for simplicity, or update if needed
                        matched_indices.append((d_idx, best_match_t_idx))


        # Create new tracks for unmatched detections
        unmatched_detections_indices = [d_idx for d_idx in range(len(detections)) if d_idx not in [m[0] for m in matched_indices]]
        for d_idx in unmatched_detections_indices:
            self.track_id_count += 1
            # [x1,y1,x2,y2,id,cls_id,name,age,hits]
            new_track = list(detections[d_idx][:4]) + [self.track_id_count] + list(detections[d_idx][5:7]) + [0, 1]
            self.tracks.append(new_track)

        # Update age for unmatched tracks and remove old tracks
        updated_tracks = []
        for track in self.tracks:
            if not any(t_idx == self.tracks.index(track) for _, t_idx in matched_indices): # If track was not matched
                track[7] += 1 # Increment age
            if track[7] <= self.max_age:
                updated_tracks.append(track)
        self.tracks = updated_tracks
        
        # Return tracks that meet min_hits criteria
        # Format: [x1, y1, x2, y2, track_id, cls_id, cls_name]
        return [track[:7] for track in self.tracks if track[8] >= self.min_hits]

