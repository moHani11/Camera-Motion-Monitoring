
from MotionDetector import MotionDetector

if __name__ == "__main__":
    detector = MotionDetector(camera_id=0, sensitivity=70, motion_threshold=10)
    detector.run()