import cv2
import threading
import time
import queue
import os
from ultralytics import YOLO
from datetime import datetime

class WebcamClassifier:
    def __init__(self):
        # Load your trained model
        self.model = YOLO('runs/classify/train/weights/best.pt')
        
        # Video settings
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Classification settings
        self.latest_prediction = ("Initializing...", 0.0)
        self.last_classification_time = 0
        self.classification_interval = 0.1  # 100ms
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_time = 0
        
        # Create output directory
        os.makedirs("Webcam_Sessions", exist_ok=True)
        
    def setup_webcam(self):
        """Initialize webcam connection"""
        print("Initializing webcam (index 0)...")
        self.cap = cv2.VideoCapture(0)  # Use default webcam
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        # Set reasonable resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test with a few reads to ensure stability
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"Webcam connected: {frame.shape[1]}x{frame.shape[0]}")
                return True
            time.sleep(0.1)
        
        print("Error: Webcam connected but cannot read frames")
        return False
    
    def start_recording(self):
        """Start recording session"""
        # First, get a stable frame to determine dimensions
        for _ in range(10):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)
        
        if not ret or frame is None:
            print("Error: Could not get stable frame for recording")
            return False
            
        height, width = frame.shape[:2]
        fps = 30.0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Webcam_Sessions/session_{timestamp}.mp4"
        
        # Use MP4 codec - try different fourcc codes if needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
        if not self.video_writer.isOpened():
            print("Error: Could not initialize video writer with any codec")
            return False
            
        self.recording = True
        print(f"Recording started: {filename}")
        print(f"Resolution: {width}x{height} at {fps} FPS")
        return True
    
    def stop_recording(self):
        """Stop recording session"""
        if self.video_writer:
            self.video_writer.release()
            self.recording = False
            print("Recording stopped and saved")
    
    def classification_worker(self):
        """Separate thread for classification"""
        while True:
            current_time = time.time()
            
            # Only classify at the specified interval
            if current_time - self.last_classification_time >= self.classification_interval:
                try:
                    # Get the most recent frame without blocking
                    if not self.frame_queue.empty():
                        # Clear queue and get only the latest frame
                        while self.frame_queue.qsize() > 1:
                            self.frame_queue.get_nowait()
                        frame = self.frame_queue.get_nowait()
                        
                        # Run classification
                        start_time = time.time()
                        results = self.model(frame)
                        self.processing_time = time.time() - start_time
                        
                        result = results[0]
                        top1 = result.probs.top1
                        top1_conf = result.probs.top1conf
                        class_names = self.model.names
                        
                        self.latest_prediction = (class_names[top1], float(top1_conf))
                        self.last_classification_time = current_time
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Classification error: {e}")
            
            time.sleep(0.01)
    
    def add_annotations(self, frame):
        """Add large, readable annotations to frame"""
        class_name, confidence = self.latest_prediction
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Dynamic font scaling - larger for better visibility
        font_scale = max(2.0, min(4.0, width / 400))
        thickness = max(3, int(font_scale * 1.2))
        
        # Main prediction text
        pred_text = f"{class_name}: {confidence:.2f}"
        text_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = 30
        text_y = text_size[1] + 50
        
        # Background for main prediction
        cv2.rectangle(frame, 
                     (text_x - 15, text_y - text_size[1] - 15),
                     (text_x + text_size[0] + 15, text_y + 15),
                     (0, 0, 0), -1)
        
        # Main prediction text
        cv2.putText(frame, pred_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Performance info
        perf_font_scale = font_scale * 0.6
        perf_thickness = max(2, int(thickness * 0.7))
        perf_text = f"FPS: {self.fps:.1f} | Process: {self.processing_time*1000:.0f}ms | Rec: {'ON' if self.recording else 'OFF'}"
        perf_y = text_y + 80
        
        perf_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, perf_font_scale, perf_thickness)[0]
        cv2.rectangle(frame,
                     (text_x - 10, perf_y - perf_size[1] - 10),
                     (text_x + perf_size[0] + 10, perf_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, perf_text, (text_x, perf_y),
                   cv2.FONT_HERSHEY_SIMPLEX, perf_font_scale, (255, 255, 255), perf_thickness)
        
        # Instructions
        instr_text = "SPACE: Stop Recording | Q: Quit"
        instr_y = height - 30
        
        instr_size = cv2.getTextSize(instr_text, cv2.FONT_HERSHEY_SIMPLEX, perf_font_scale, perf_thickness)[0]
        cv2.rectangle(frame,
                     (text_x - 10, instr_y - instr_size[1] - 10),
                     (text_x + instr_size[0] + 10, instr_y + 5),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, instr_text, (text_x, instr_y),
                   cv2.FONT_HERSHEY_SIMPLEX, perf_font_scale, (255, 255, 255), perf_thickness)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        """Main application loop"""
        if not self.setup_webcam():
            return
        
        # Start classification thread
        classification_thread = threading.Thread(target=self.classification_worker, daemon=True)
        classification_thread.start()
        
        # Start recording immediately
        if not self.start_recording():
            print("Failed to start recording, continuing without recording...")
            # Continue without recording rather than exiting
        
        print("Webcam Classification Started!")
        print("Video: Targeting 30 FPS")
        print("Classification: Every 100ms")
        print("Recording: Started automatically")
        print("Controls: SPACE to stop recording | Q to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam")
                    break
                
                # Calculate FPS
                self.calculate_fps()
                
                # Add current frame to classification queue (non-blocking)
                if self.frame_queue.qsize() < 2:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # Add annotations to frame
                annotated_frame = self.add_annotations(frame)
                
                # Write to video file if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Webcam - Live Classification', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # SPACE to stop recording
                    if self.recording:
                        self.stop_recording()
                elif key == ord('q'):  # Q to quit
                    break
                elif key == ord('r'):  # R to restart recording
                    if not self.recording:
                        self.start_recording()
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

# Run the application
if __name__ == "__main__":
    classifier = WebcamClassifier()
    classifier.run()