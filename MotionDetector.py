import numpy as np
import cv2 as cv
import os
from collections import deque
from datetime import datetime

class MotionDetector:
    
    def __init__(self, camera_id=0, sensitivity=70, motion_threshold=10):
        self.camera_id = camera_id
        self.sensitivity = sensitivity
        self.motion_threshold = motion_threshold
        
        # keep track of motion over last 30 frames
        self.motion_history = deque(maxlen=30)
        # trail showing where motion happened
        self.max_motion_points = deque(maxlen=100)
        
        self.is_recording = False
        self.video_writer = None
        
        self.frame_count = 0
        self.motion_detected_frames = 0
        
    def normalize_magnitude(self, magnitude, min_val, max_val):
        return magnitude * (max_val - min_val) / self.sensitivity
    
    def setup_video_writer(self, frame_shape, fps=20.0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_recording_{timestamp}.avi"
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        height, width = frame_shape[:2]
        self.video_writer = cv.VideoWriter(filename, fourcc, fps, (width, height))
        print(f"Recording started: {filename}")
    #------------------------------------------------------------
        
    def draw_motion_info(self, frame, mag, max_pos, avg_motion):
        height, width = frame.shape[:2]
        
        # draw the trail of motion points with fading colors
        for i, point in enumerate(self.max_motion_points):
            alpha = (i + 1) / len(self.max_motion_points)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv.circle(frame, point, 3, color, -1)
        
        # circle around the current hotspot
        cv.circle(frame, (max_pos[1], max_pos[0]), 20, (0, 255, 0), 2)
        # cv.circle(frame, (max_pos[1], max_pos[0]), 20, (0, 255, 0), 2)
        # cv.circle(frame, (max_pos[1], max_pos[0]), 20, (0, 255, 0), 2)
        cv.circle(frame, (max_pos[1], max_pos[0]), 3, (255, 255, 255), -1)
        
        max_mag = mag[max_pos[0], max_pos[1]]
        
        # black box for info text
        cv.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
        cv.rectangle(frame, (10, 10), (300, 150), (255, 255, 255), 2)
        
        # put all the stats on screen
        info = [
            f"Frame: {self.frame_count}",
            f"Max Motion: {max_mag:.2f}",
            f"Avg Motion: {avg_motion:.2f}",
            f"Motion %: {self.get_motion_percentage():.1f}%",
            f"Recording: {'ON' if self.is_recording else 'OFF'}"
        ]
        
        for i, text in enumerate(info):
            cv.putText(frame, text, (20, 35 + i * 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # big warning when motion is detected
        if max_mag > self.motion_threshold:
            cv.putText(frame, "MOTION DETECTED!", (width - 250, 40),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def draw_motion_graph(self, frame):
        if len(self.motion_history) < 2:
            return frame
        
        height, width = frame.shape[:2]
        graph_height = 100
        graph_width = 300
        graph_x = width - graph_width - 10
        graph_y = height - graph_height - 10
        
        # black background for graph
        cv.rectangle(frame, (graph_x, graph_y), 
                    (graph_x + graph_width, graph_y + graph_height),
                    (0, 0, 0), -1)
        cv.rectangle(frame, (graph_x, graph_y),
                    (graph_x + graph_width, graph_y + graph_height),
                    (255, 255, 255), 1)
        
        # draw the motion values as a line graph
        history = list(self.motion_history)
        max_val = max(history) if max(history) > 0 else 1
        
        for i in range(len(history) - 1):
            x1 = graph_x + int(i * graph_width / len(history))
            y1 = graph_y + graph_height - int(history[i] / max_val * graph_height)
            x2 = graph_x + int((i + 1) * graph_width / len(history))
            y2 = graph_y + graph_height - int(history[i + 1] / max_val * graph_height)
            cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # red line showing the threshold
        threshold_y = graph_y + graph_height - int(self.motion_threshold / max_val * graph_height)
        cv.line(frame, (graph_x, threshold_y), 
               (graph_x + graph_width, threshold_y), (0, 0, 255), 1)
        
        return frame
    
    def get_motion_percentage(self):
        if self.frame_count == 0:
            return 0.0
        return (self.motion_detected_frames / self.frame_count) * 100
    #------------------------------------------------------------

    def run(self):
        die_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(die_path)
        
        cap = cv.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        ret, frame1 = cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            return
        
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        
        print("\n=== Controls ===")
        print("q - Quit")
        print("s - Save current frame")
        print("r - Toggle recording")
        print("+/- - Adjust sensitivity")
        print("c - Clear trail")
        print("================\n")
        
        while True:
            ret, frame2 = cap.read()
            if not ret:
                print('No frames grabbed!')
                break
            
            self.frame_count += 1
            
            # get optical flow between frames
            next_frame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            
            # convert flow to magnitude and angle
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            mag = self.normalize_magnitude(mag, 0, 255)
            
            # find where motion is strongest
            max_index = np.argmax(mag)
            max_pos = np.unravel_index(max_index, mag.shape)
            max_mag = mag[max_pos[0], max_pos[1]]
            
            # track motion over time
            avg_motion = np.mean(mag)
            self.motion_history.append(avg_motion)
            
            if max_mag > self.motion_threshold:
                self.motion_detected_frames += 1
                self.max_motion_points.append((max_pos[1], max_pos[0]))
            
            # make the flow visualization colorful
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = np.clip(mag, 0, 255).astype(np.uint8)
            bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            
            # add all the overlays
            display_frame = frame2.copy()
            display_frame = self.draw_motion_info(display_frame, mag, max_pos, avg_motion)
            display_frame = self.draw_motion_graph(display_frame)
            
            cv.imshow('Motion Detection', display_frame)
            cv.imshow('Optical Flow', bgr_flow)
            
            # save video if recording
            if self.is_recording:
                if self.video_writer is None:
                    self.setup_video_writer(display_frame.shape)
                self.video_writer.write(display_frame)
            
            # handle keyboard input
            key = cv.waitKey(30) & 0xff
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv.imwrite(f'motion_frame_{timestamp}.png', display_frame)
                cv.imwrite(f'optical_flow_{timestamp}.png', bgr_flow)
                print(f"Saved images: {timestamp}")
            elif key == ord('r'):# me4 3aref me4 4a8ala leih
                self.is_recording = not self.is_recording
                if not self.is_recording and self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    print("Recording stopped")
            elif key == ord('+'):
                self.sensitivity = max(10, self.sensitivity - 5)
                print(f"Sensitivity: {self.sensitivity}")
            elif key == ord('-'):
                self.sensitivity = min(200, self.sensitivity + 5)
                print(f"Sensitivity: {self.sensitivity}")
            elif key == ord('c'):
                self.max_motion_points.clear()
                print("Trail cleared")
            
            prvs = next_frame
        
        # print some stats before closing
        print(f"\nTotal frames: {self.frame_count}")
        print(f"Motion detected: {self.motion_detected_frames} ({self.get_motion_percentage():.1f}%)")
        
        if self.video_writer is not None:
            self.video_writer.release()
        cap.release()
        cv.destroyAllWindows()
