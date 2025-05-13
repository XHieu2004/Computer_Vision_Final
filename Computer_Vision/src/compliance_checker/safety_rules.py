class SafetyComplianceChecker:
    def __init__(self, require_helmet=True, require_vest=True):
        self.require_helmet = require_helmet
        self.require_vest = require_vest

    def check_ppe_compliance(self, person_ppe_status):
        """
        Checks PPE compliance for each person and logs violations.
        :param person_ppe_status: Dictionary from PPEAssociator.
            Example: {person_id: {'helmet_status': 'helmet'/'no-helmet'/'unknown',
                                     'vest_status': 'vest'/'no-vest'/'unknown',
                                     'bbox': person_box}}
        :return: Dictionary of violations.
            Example: {person_id: {'violations': ["No Helmet Detected"], 'bbox': person_box}}
        """
        violations = {}
        violations_found_in_frame = False
        for person_id, status in person_ppe_status.items():
            person_violations = []
            
            # Check for helmet
            if self.require_helmet:
                if status['helmet_status'] == 'no-helmet':
                    person_violations.append("No Helmet Detected")
                elif status['helmet_status'] == 'unknown':
                    # Decide if 'unknown' is a violation or just needs review
                    person_violations.append("Helmet Status Unknown") 

            # Check for vest
            if self.require_vest:
                if status['vest_status'] == 'no-vest':
                    person_violations.append("No Vest Detected")
                elif status['vest_status'] == 'unknown':
                    # Decide if 'unknown' is a violation
                    person_violations.append("Vest Status Unknown")

            if person_violations:
                violations[person_id] = {
                    'violations': person_violations,
                    'bbox': status['bbox']  # Include bbox for drawing or logging
                }
                # Log the violation for this person
                print(f"[VIOLATION LOG] Person ID: {person_id}")
                for pv in person_violations:
                    print(f"  - {pv}")
                violations_found_in_frame = True
        
        if not violations_found_in_frame:
            print("[COMPLIANCE LOG] No PPE violations detected in this frame.")
            
        return violations
