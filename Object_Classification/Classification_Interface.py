import cv2
import threading
import time
import queue
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import json

model_path = 'Object_Classification/runs/classify/train/weights/best.pt'

class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, width=120, height=40, 
                 color="#4a7abc", hover_color="#3a6aac", text_color="white"):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, bg=parent.cget('bg'))
        self.command = command
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        
        # Create rounded rectangle
        self.rect = self.create_round_rect(2, 2, width-2, height-2, 
                                         radius=15, fill=color, outline="")
        
        # Add text
        self.text_id = self.create_text(width//2, height//2, text=text,
                                       fill=text_color, font=("Arial", 10, "bold"))
        
        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def create_round_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                 x2-radius, y1,
                 x2, y1,
                 x2, y1+radius,
                 x2, y2-radius,
                 x2, y2,
                 x2-radius, y2,
                 x1+radius, y2,
                 x1, y2,
                 x1, y2-radius,
                 x1, y1+radius,
                 x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def on_enter(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)
        
    def on_leave(self, event):
        self.itemconfig(self.rect, fill=self.color)
        
    def on_click(self, event):
        self.command()
        
    def update_color(self, new_color):
        self.color = new_color
        self.itemconfig(self.rect, fill=new_color)
        
    def update_text(self, new_text):
        self.itemconfig(self.text_id, text=new_text)

class StatisticsCollector:
    """Collects and analyzes classification statistics"""
    def __init__(self, class_names):
        self.class_names = class_names
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all statistics"""
        self.classification_history = []
        self.class_counts = {name: 0 for name in self.class_names.values()}
        self.confidence_scores = {name: [] for name in self.class_names.values()}
        self.frame_count = 0
        self.start_time = datetime.now()
        
    def add_classification(self, classifications, processing_time):
        """Add classification results to statistics"""
        self.frame_count += 1
        timestamp = datetime.now()
        
        for classification in classifications:
            class_name = classification['class_name']
            confidence = classification['confidence']
            
            # Update class counts
            self.class_counts[class_name] += 1
            
            # Store confidence scores
            self.confidence_scores[class_name].append(confidence)
            
            # Add to history
            self.classification_history.append({
                'timestamp': timestamp,
                'class_name': class_name,
                'confidence': confidence,
                'processing_time': processing_time
            })
    
    def generate_pdf_report(self, filename=None):
        """Generate PDF report with statistics and charts"""
        if not filename:
            filename = f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
        with PdfPages(filename) as pdf:
            # Create summary page
            self._create_summary_page(pdf)
            
            # Create class distribution chart
            self._create_class_distribution_chart(pdf)
            
            # Create confidence histograms
            self._create_confidence_histograms(pdf)
            
            # Create temporal analysis
            self._create_temporal_analysis(pdf)
            
        return filename
    
    def _create_summary_page(self, pdf):
        """Create summary page for PDF report"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Classification Analysis Report', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Session info
        session_duration = datetime.now() - self.start_time
        total_classifications = sum(self.class_counts.values())
        
        info_text = (
            f"Session Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Session Duration: {str(session_duration).split('.')[0]}\n"
            f"Total Frames Processed: {self.frame_count}\n"
            f"Total Classifications: {total_classifications}\n"
            f"Classification Rate: {total_classifications/max(1, self.frame_count):.2f} classifications/frame\n"
        )
        
        ax.text(0.1, 0.8, info_text, fontsize=12, va='top', linespacing=1.5)
        
        # Class summary
        class_summary = "Class Distribution:\n"
        for class_name, count in sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                class_summary += f"  {class_name}: {count} ({count/max(1, total_classifications)*100:.1f}%)\n"
        
        ax.text(0.1, 0.5, class_summary, fontsize=10, va='top', linespacing=1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_class_distribution_chart(self, pdf):
        """Create class distribution pie chart"""
        # Filter out classes with no classifications
        active_classes = {k: v for k, v in self.class_counts.items() if v > 0}
        
        if not active_classes:
            return
            
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(active_classes)))
        
        wedges, texts, autotexts = ax.pie(active_classes.values(), labels=active_classes.keys(), 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        ax.set_title('Class Distribution', fontsize=16, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_histograms(self, pdf):
        """Create confidence score histograms"""
        # Get classes with sufficient data
        active_classes = {k: v for k, v in self.confidence_scores.items() if len(v) > 0}
        
        if not active_classes:
            return
            
        n_classes = len(active_classes)
        n_cols = 2
        n_rows = (n_classes + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
        if n_classes == 1:
            axes = [axes]
        elif n_rows > 1:
            axes = axes.flatten()
        
        for idx, (class_name, confidences) in enumerate(active_classes.items()):
            if idx < len(axes):
                ax = axes[idx]
                ax.hist(confidences, bins=20, alpha=0.7, color=f'C{idx}', edgecolor='black')
                ax.set_title(f'{class_name} Confidence', fontweight='bold')
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_conf = np.mean(confidences)
                ax.axvline(mean_conf, color='red', linestyle='--', 
                          label=f'Mean: {mean_conf:.3f}')
                ax.legend()
        
        # Hide empty subplots
        for idx in range(len(active_classes), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_analysis(self, pdf):
        """Create temporal analysis of classifications"""
        if len(self.classification_history) < 2:
            return
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.classification_history)
        df['time_seconds'] = (df['timestamp'] - self.start_time).dt.total_seconds()
        
        # Create temporal distribution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Classification frequency over time
        time_bins = np.linspace(0, df['time_seconds'].max(), 20)
        ax1.hist(df['time_seconds'], bins=time_bins, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Classification Count')
        ax1.set_title('Classification Frequency Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Processing time distribution
        ax2.hist(df['processing_time'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Processing Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Processing Time Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

class LiveClassifierGUI:
    def __init__(self, model, parent, stats):
        self.model = model
        self.parent = parent
        self.stats = stats
        
        # Video settings
        self.cap = None
        self.recording = True  # Start recording automatically
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        
        # Classification settings
        self.latest_classification = None
        self.last_classification_time = 0
        self.classification_interval = 0.1  # 100ms
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_time = 0
        
        # Create output directory
        os.makedirs("Webcam_Classifications", exist_ok=True)
        
        # Colors for different confidence levels
        self.confidence_colors = {
            'high': (0, 255, 0),    # Green
            'medium': (255, 255, 0), # Yellow
            'low': (255, 0, 0)      # Red
        }
    
    def setup_webcam(self):
        """Initialize webcam connection"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.parent.update_status("Error: Could not open webcam", "#e74c3c")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test webcam
        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.parent.update_status("Webcam connected - Starting classification...")
                return True
            time.sleep(0.1)
        
        self.parent.update_status("Error: Webcam connected but cannot read frames", "#e74c3c")
        return False
    
    def start_recording(self):
        """Start recording session"""
        if not self.cap:
            return False
            
        # Get frame for dimensions
        for _ in range(10):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)
        
        if not ret or frame is None:
            return False
            
        height, width = frame.shape[:2]
        fps = 30.0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Webcam_Classifications/session_{timestamp}.mp4"
        
        # Try different codecs
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
        if not self.video_writer.isOpened():
            return False
            
        self.recording = True
        self.parent.update_record_button(True)
        return True
    
    def stop_recording(self):
        """Stop recording session"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.parent.update_record_button(False)
    
    def classification_worker(self):
        """Separate thread for image classification"""
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_classification_time >= self.classification_interval:
                try:
                    if not self.frame_queue.empty():
                        # Get only the latest frame
                        while self.frame_queue.qsize() > 1:
                            self.frame_queue.get_nowait()
                        frame = self.frame_queue.get_nowait()
                        
                        # Run classification with current settings
                        start_time = time.time()
                        results = self.model(frame, conf=self.parent.confidence_threshold, verbose=False)
                        self.processing_time = time.time() - start_time
                        
                        # Process results - classification models return different structure
                        classifications = []
                        result = results[0]
                        
                        if result.probs is not None:
                            # Get top prediction
                            top1_idx = result.probs.top1
                            top1_conf = result.probs.top1conf.item()
                            class_name = self.model.names[top1_idx]
                            
                            # Apply class filter
                            if self.parent.selected_classes.get(class_name, True):
                                classifications.append({
                                    'class_name': class_name,
                                    'confidence': float(top1_conf),
                                    'class_id': int(top1_idx)
                                })
                        
                        self.latest_classification = classifications[0] if classifications else None
                        self.last_classification_time = current_time
                        
                        # Add to statistics
                        if self.stats and classifications:
                            self.stats.add_classification(classifications, self.processing_time)
                        
                except Exception as e:
                    print(f"Classification error: {e}")
            
            time.sleep(0.01)
    
    def draw_classification(self, frame):
        """Draw classification results on frame"""
        if self.latest_classification:
            class_name = self.latest_classification['class_name']
            confidence = self.latest_classification['confidence']
            
            # Determine color based on confidence
            if confidence >= 0.7:
                color = self.confidence_colors['high']
            elif confidence >= 0.4:
                color = self.confidence_colors['medium']
            else:
                color = self.confidence_colors['low']
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Create classification text - moved to lower position to avoid overlap
            classification_text = f"Class: {class_name}"
            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Calculate text size
            font_scale = min(width / 800, height / 600) * 1.2
            thickness = max(2, int(font_scale * 1.5))
            
            # Position classification text at top center (but lower to avoid performance info)
            (text_width, text_height), baseline = cv2.getTextSize(
                classification_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            text_x = (width - text_width) // 2
            text_y = text_height + 60  # Increased from 20 to 60 to move below performance info
            
            # Draw background for classification text
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + 10),
                         color, -1)
            
            # Draw classification text
            cv2.putText(frame, classification_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Position confidence text below classification
            (conf_width, conf_height), _ = cv2.getTextSize(
                confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness - 1
            )
            
            conf_x = (width - conf_width) // 2
            conf_y = text_y + conf_height + 25
            
            # Draw background for confidence text
            cv2.rectangle(frame,
                         (conf_x - 10, conf_y - conf_height - 10),
                         (conf_x + conf_width + 10, conf_y + 10),
                         color, -1)
            
            # Draw confidence text
            cv2.putText(frame, confidence_text, (conf_x, conf_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness - 1)
            
            # Draw confidence bar at bottom
            bar_width = int(width * 0.6)
            bar_height = 25
            bar_x = (width - bar_width) // 2
            bar_y = height - 50
            
            # Draw bar background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Draw confidence level
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
            # Draw bar border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            # Draw confidence percentage text
            percent_text = f"{confidence * 100:.1f}%"
            (percent_width, percent_height), _ = cv2.getTextSize(
                percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            percent_x = bar_x + (bar_width - percent_width) // 2
            percent_y = bar_y + bar_height // 2 + percent_height // 2
            
            cv2.putText(frame, percent_text, (percent_x, percent_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def add_annotations(self, frame):
        """Add performance info to frame - moved to top left corner"""
        height, width = frame.shape[:2]
        
        # Dynamic font scaling
        font_scale = max(0.6, min(1.0, width / 1000))  # Smaller font to fit in corner
        thickness = max(1, int(font_scale * 1.5))
        
        # Performance info (top left corner, compact)
        current_class = self.latest_classification['class_name'] if self.latest_classification else "None"
        perf_text = f"FPS: {self.fps:.1f} | {self.processing_time*1000:.0f}ms | Rec: {'ON' if self.recording else 'OFF'}"
        
        # Background for performance info - smaller and in corner
        perf_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.rectangle(frame,
                     (5, 5),  # Moved closer to corner
                     (10 + perf_size[0], 10 + perf_size[1]),
                     (0, 0, 0), -1)
        
        # Performance text
        cv2.putText(frame, perf_text, (8, 8 + perf_size[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def start(self):
        """Start live classification"""
        if not self.setup_webcam():
            return
        
        self.running = True
        
        # Start recording automatically
        if not self.start_recording():
            self.parent.update_status("Failed to start recording", "#e67e22")
        
        # Start classification thread
        self.classification_thread = threading.Thread(target=self.classification_worker, daemon=True)
        self.classification_thread.start()
        
        # Start main loop
        self.main_loop()
    
    def main_loop(self):
        """Main classification loop"""
        if not self.running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.parent.update_status("Error: Could not read frame from webcam", "#e74c3c")
            return
        
        # Calculate FPS
        self.calculate_fps()
        
        # Add frame to classification queue
        if self.frame_queue.qsize() < 2:
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
        
        # Draw classifications and annotations
        frame_with_classification = self.draw_classification(frame)
        annotated_frame = self.add_annotations(frame_with_classification)
        
        # Write to video if recording
        if self.recording and self.video_writer:
            self.video_writer.write(annotated_frame)
        
        # Update GUI
        self.parent.update_video_frame(annotated_frame)
        
        # Update status
        current_class = self.latest_classification['class_name'] if self.latest_classification else "None"
        current_confidence = self.latest_classification['confidence'] if self.latest_classification else 0
        status_text = f"FPS: {self.fps:.1f} | Class: {current_class} ({current_confidence:.2f}) | Recording: {'ON' if self.recording else 'OFF'}"
        self.parent.update_status(status_text)
        
        # Continue loop
        if self.running:
            self.parent.root.after(10, self.main_loop)
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def stop(self):
        """Stop live classification"""
        self.running = False
        if self.recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()

class MainApplication:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 Classification - Enhanced")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(True, True)
        
        # Load the model once at startup
        self.model_path = model_path
        try:
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names
            print(f"Classification model loaded successfully with {len(self.class_names)} classes")
            
            # Initialize statistics collector
            self.stats = StatisticsCollector(self.class_names)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}")
            self.model = None
            self.class_names = {}
            self.stats = None
        
        # Classification settings with defaults
        self.confidence_threshold = 0.25
        self.selected_classes = {name: True for name in self.class_names.values()} if self.class_names else {}
        
        self.current_screen = None
        self.show_main_menu()
        
        # Save statistics on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        """Save PDF report when application closes"""
        if hasattr(self, 'stats') and self.stats:
            try:
                report_path = self.stats.generate_pdf_report()
                print(f"PDF report saved to: {report_path}")
            except Exception as e:
                print(f"Error generating PDF report: {e}")
        self.root.quit()
        
    def clear_screen(self):
        """Clear all widgets from current screen"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_main_menu(self):
        """Display the main menu screen"""
        self.clear_screen()
        self.current_screen = "main"
        self.root.configure(bg="#2c3e50")
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(expand=True, fill="both", padx=50, pady=50)
        
        # Title
        title_frame = tk.Frame(main_frame, bg="#2c3e50")
        title_frame.pack(pady=40)
        
        title_label = tk.Label(title_frame, text="YOLOv8 Classification", 
                              font=("Arial", 28, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(title_frame, text="Enhanced Classification System", 
                                 font=("Arial", 14), fg="#ecf0f1", bg="#2c3e50")
        subtitle_label.pack()
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg="#2c3e50")
        button_frame.pack(pady=50)
        
        # Classify Images button
        classify_btn = ModernButton(button_frame, "Classify Images from Folder", 
                                 command=self.show_image_classification_screen,
                                 width=300, height=60, color="#27ae60", hover_color="#219653")
        classify_btn.pack(pady=20)
        
        # Live Classification button
        live_btn = ModernButton(button_frame, "Live Webcam Classification", 
                              command=self.show_live_classification_screen,
                              width=300, height=60, color="#e74c3c", hover_color="#c0392b")
        live_btn.pack(pady=20)
        
        # Statistics Dashboard button
        stats_btn = ModernButton(button_frame, "Statistics Dashboard", 
                               command=self.show_statistics_dashboard,
                               width=300, height=60, color="#3498db", hover_color="#2980b9")
        stats_btn.pack(pady=20)
        
        # Exit button
        exit_frame = tk.Frame(main_frame, bg="#2c3e50")
        exit_frame.pack(pady=30)
        
        exit_btn = ModernButton(exit_frame, "Exit", command=self.on_closing,
                              width=150, height=45, color="#95a5a6", hover_color="#7f8c8d")
        exit_btn.pack()
    
    def show_statistics_dashboard(self):
        """Display the statistics dashboard"""
        if not hasattr(self, 'stats') or not self.stats:
            messagebox.showinfo("Info", "No statistics data available yet.")
            return
            
        self.clear_screen()
        self.current_screen = "statistics"
        self.root.configure(bg="#34495e")
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#34495e")
        main_frame.pack(expand=True, fill="both", padx=40, pady=30)
        
        # Title
        title_label = tk.Label(main_frame, text="Statistics Dashboard", 
                              font=("Arial", 24, "bold"), fg="white", bg="#34495e")
        title_label.pack(pady=30)
        
        # Statistics summary
        stats_frame = tk.Frame(main_frame, bg="#34495e")
        stats_frame.pack(fill="x", padx=20, pady=20)
        
        # Total classifications
        total_classifications = sum(self.stats.class_counts.values())
        total_frames = self.stats.frame_count
        
        stats_text = (
            f"Session Duration: {str(datetime.now() - self.stats.start_time).split('.')[0]}\n"
            f"Total Frames Processed: {total_frames}\n"
            f"Total Classifications: {total_classifications}\n"
            f"Classification Rate: {total_classifications/max(1, total_frames):.2f} classifications/frame\n"
        )
        
        stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 12), 
                              fg="white", bg="#34495e", justify="left")
        stats_label.pack(anchor="w")
        
        # Class distribution
        class_frame = tk.Frame(main_frame, bg="#34495e")
        class_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(class_frame, text="Class Distribution:", font=("Arial", 14, "bold"), 
                fg="white", bg="#34495e").pack(anchor="w")
        
        # Create scrollable frame for class list
        class_canvas = tk.Canvas(class_frame, bg="#34495e", highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(class_frame, orient="vertical", command=class_canvas.yview)
        scrollable_frame = tk.Frame(class_canvas, bg="#34495e")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: class_canvas.configure(scrollregion=class_canvas.bbox("all"))
        )
        
        class_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        class_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add class items
        for class_name, count in sorted(self.stats.class_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                class_item = tk.Frame(scrollable_frame, bg="#34495e")
                class_item.pack(fill="x", pady=2)
                
                class_label = tk.Label(class_item, 
                                      text=f"{class_name}: {count} ({count/max(1, total_classifications)*100:.1f}%)",
                                      font=("Arial", 10), fg="white", bg="#34495e")
                class_label.pack(side="left")
        
        class_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg="#34495e")
        button_frame.pack(pady=20)
        
        # Export PDF button
        pdf_btn = ModernButton(button_frame, "Export PDF Report", 
                             command=self.export_pdf_report,
                             width=180, height=45, color="#3498db", hover_color="#2980b9")
        pdf_btn.pack(side="left", padx=10)
        
        # Reset stats button
        reset_btn = ModernButton(button_frame, "Reset Statistics", 
                               command=self.reset_statistics,
                               width=150, height=45, color="#e74c3c", hover_color="#c0392b")
        reset_btn.pack(side="left", padx=10)
        
        # Back button
        back_btn = ModernButton(button_frame, "Back to Main", 
                              command=self.show_main_menu,
                              width=150, height=45, color="#95a5a6", hover_color="#7f8c8d")
        back_btn.pack(side="left", padx=10)
    
    def export_pdf_report(self):
        """Export statistics as PDF"""
        try:
            report_path = self.stats.generate_pdf_report()
            messagebox.showinfo("Success", f"PDF report exported to:\n{report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {e}")
    
    def reset_statistics(self):
        """Reset all statistics"""
        if hasattr(self, 'stats'):
            self.stats.reset_stats()
            messagebox.showinfo("Success", "Statistics reset successfully!")
            self.show_statistics_dashboard()

    def show_image_classification_screen(self):
        """Display the image classification screen"""
        self.clear_screen()
        self.current_screen = "image_classification"
        self.root.configure(bg="#34495e")
        
        # Create a main frame that fills the window
        main_frame = tk.Frame(self.root, bg="#34495e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Image Classification", 
                              font=("Arial", 24, "bold"), fg="white", bg="#34495e")
        title_label.pack(pady=20)
        
        # Compact settings frame
        settings_frame = tk.LabelFrame(main_frame, text="Classification Settings", 
                                      font=("Arial", 12, "bold"), fg="white", bg="#34495e",
                                      padx=10, pady=10)
        settings_frame.pack(fill="x", pady=10)
        
        # Confidence threshold - Compact row
        conf_frame = tk.Frame(settings_frame, bg="#34495e")
        conf_frame.pack(fill="x", pady=5)
        
        tk.Label(conf_frame, text="Confidence:", font=("Arial", 10, "bold"), 
                fg="white", bg="#34495e").pack(side="left")
        
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        conf_scale = tk.Scale(conf_frame, from_=0.1, to=0.9, resolution=0.05, 
                             orient="horizontal", variable=self.confidence_var,
                             command=self.update_confidence_threshold,
                             length=200, showvalue=True,
                             bg="#34495e", fg="white", highlightbackground="#34495e")
        conf_scale.pack(side="left", padx=10)
        
        # Input/Output section
        io_frame = tk.LabelFrame(main_frame, text="Input/Output", 
                                font=("Arial", 12, "bold"), fg="white", bg="#34495e",
                                padx=10, pady=10)
        io_frame.pack(fill="x", pady=10)
        
        # Input selection buttons
        input_buttons_frame = tk.Frame(io_frame, bg="#34495e")
        input_buttons_frame.pack(fill="x", pady=5)
        
        tk.Label(input_buttons_frame, text="Select Input:", font=("Arial", 10, "bold"),
                fg="white", bg="#34495e").pack(anchor="w")
        
        buttons_subframe = tk.Frame(input_buttons_frame, bg="#34495e")
        buttons_subframe.pack(fill="x", pady=5)
        
        # Folder selection button
        folder_btn = ModernButton(buttons_subframe, "Select Folder", 
                                 command=self.browse_input_folder,
                                 width=120, height=35, color="#3498db", hover_color="#2980b9")
        folder_btn.pack(side="left", padx=(0, 10))
        
        # Individual image selection button
        image_btn = ModernButton(buttons_subframe, "Select Images", 
                                command=self.browse_input_images,
                                width=120, height=35, color="#9b59b6", hover_color="#8e44ad")
        image_btn.pack(side="left", padx=(0, 10))
        
        # Selected path display
        self.input_path_var = tk.StringVar()
        input_entry = tk.Entry(io_frame, textvariable=self.input_path_var, 
                              font=("Arial", 9), bg="#ecf0f1", fg="#2c3e50", state="readonly")
        input_entry.pack(fill="x", pady=5)
        
        # Output folder selection
        output_frame = tk.Frame(io_frame, bg="#34495e")
        output_frame.pack(fill="x", pady=5)
        
        tk.Label(output_frame, text="Output Folder:", font=("Arial", 10, "bold"),
                fg="white", bg="#34495e").pack(anchor="w")
        
        output_buttons_frame = tk.Frame(output_frame, bg="#34495e")
        output_buttons_frame.pack(fill="x", pady=2)
        
        # Output folder entry and browse button
        self.output_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "classification_results"))
        output_entry = tk.Entry(output_buttons_frame, textvariable=self.output_folder_var, 
                               font=("Arial", 9), bg="#ecf0f1", fg="#2c3e50")
        output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        output_browse_btn = ModernButton(output_buttons_frame, "Browse", 
                                       command=self.browse_output_folder,
                                       width=80, height=30, color="#3498db", hover_color="#2980b9")
        output_browse_btn.pack(side="right")
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg="#34495e")
        button_frame.pack(pady=20)
        
        classify_btn = ModernButton(button_frame, "Classify Images", 
                              command=self.classify_images,
                              width=160, height=45, color="#27ae60", hover_color="#219653")
        classify_btn.pack(side="left", padx=10)
        
        back_btn = ModernButton(button_frame, "Back to Main", 
                              command=self.show_main_menu,
                              width=160, height=45, color="#95a5a6", hover_color="#7f8c8d")
        back_btn.pack(side="left", padx=10)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="", font=("Arial", 10), 
                                   fg="#3498db", bg="#34495e")
        self.status_label.pack(pady=10)
        
        # Store selected images
        self.selected_images = []
    
    def update_confidence_threshold(self, value):
        """Update confidence threshold from slider"""
        self.confidence_threshold = float(value)
    
    def browse_input_folder(self):
        """Open file dialog to select input folder"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder_path:
            self.input_path_var.set(folder_path)
            self.selected_images = []  # Clear individual image selection
    
    def browse_input_images(self):
        """Open file dialog to select individual images"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if file_paths:
            # Display the number of selected images
            self.input_path_var.set(f"{len(file_paths)} images selected")
            self.selected_images = list(file_paths)
    
    def browse_output_folder(self):
        """Open file dialog to select output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder for Classification Results")
        if folder_path:
            self.output_folder_var.set(folder_path)
    
    def classify_images(self):
        """Process selected images or folder"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        output_folder = self.output_folder_var.get()
        
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder!")
            return
        
        # Determine input type
        if self.selected_images:
            # Process individual images
            image_paths = self.selected_images
        else:
            # Process folder
            input_folder = self.input_path_var.get()
            if not input_folder:
                messagebox.showerror("Error", "Please select an input folder or images!")
                return
            
            # Get all image files from folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
        if not image_paths:
            messagebox.showwarning("Warning", "No images found!")
            return
        
        # Update status
        self.status_label.config(text="Classifying images...")
        self.root.update()
        
        try:
            # Run image processing in a thread to avoid freezing GUI
            thread = threading.Thread(target=self._classify_images_thread, 
                                    args=(image_paths, output_folder))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")
            self.status_label.config(text="")
    
    def _classify_images_thread(self, image_paths, output_folder):
        """Thread function for image classification"""
        try:
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each image
            processed_count = 0
            for i, image_path in enumerate(image_paths):
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Run inference with current confidence threshold
                results = self.model(image, conf=self.confidence_threshold, verbose=False)
                
                # Draw classification results
                classified_image = self.draw_classification_on_image(image, results[0])
                
                # Save classified image
                output_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(output_path, classified_image)
                processed_count += 1
            
            # Show completion message with full path
            full_output_path = os.path.abspath(output_folder)
            self.root.after(0, lambda: messagebox.showinfo("Complete", 
                f"Classified {processed_count} images!\n\nSaved to:\n{full_output_path}"))
            self.root.after(0, lambda: self.status_label.config(text=""))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Classification failed: {e}"))
            self.root.after(0, lambda: self.status_label.config(text=""))
    
    def draw_classification_on_image(self, image, result):
        """Draw classification results on a single image"""
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        if result.probs is not None:
            # Get top prediction
            top1_idx = result.probs.top1
            top1_conf = result.probs.top1conf.item()
            class_name = self.model.names[top1_idx]
            
            # Determine color based on confidence
            if top1_conf >= 0.7:
                color = (0, 255, 0)    # Green
            elif top1_conf >= 0.4:
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)    # Red
            
            # Create classification text
            classification_text = f"Class: {class_name}"
            confidence_text = f"Confidence: {top1_conf:.2f}"
            
            # Calculate text size
            font_scale = min(width / 800, height / 600) * 1.5
            thickness = max(2, int(font_scale * 1.5))
            
            # Position classification text at top center
            (text_width, text_height), baseline = cv2.getTextSize(
                classification_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            text_x = (width - text_width) // 2
            text_y = text_height + 40
            
            # Draw background for classification text
            cv2.rectangle(annotated_image, 
                         (text_x - 15, text_y - text_height - 15),
                         (text_x + text_width + 15, text_y + 15),
                         color, -1)
            
            # Draw classification text
            cv2.putText(annotated_image, classification_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Position confidence text below classification
            (conf_width, conf_height), _ = cv2.getTextSize(
                confidence_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, thickness - 1
            )
            
            conf_x = (width - conf_width) // 2
            conf_y = text_y + conf_height + 30
            
            # Draw background for confidence text
            cv2.rectangle(annotated_image,
                         (conf_x - 10, conf_y - conf_height - 10),
                         (conf_x + conf_width + 10, conf_y + 10),
                         color, -1)
            
            # Draw confidence text
            cv2.putText(annotated_image, confidence_text, (conf_x, conf_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (255, 255, 255), thickness - 1)
            
            # Draw confidence bar at bottom
            bar_width = int(width * 0.7)
            bar_height = 35
            bar_x = (width - bar_width) // 2
            bar_y = height - 60
            
            # Draw bar background
            cv2.rectangle(annotated_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Draw confidence level
            confidence_width = int(bar_width * top1_conf)
            cv2.rectangle(annotated_image, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
            # Draw bar border
            cv2.rectangle(annotated_image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 3)
            
            # Draw confidence percentage text
            percent_text = f"{top1_conf * 100:.1f}%"
            (percent_width, percent_height), _ = cv2.getTextSize(
                percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            percent_x = bar_x + (bar_width - percent_width) // 2
            percent_y = bar_y + bar_height // 2 + percent_height // 2
            
            cv2.putText(annotated_image, percent_text, (percent_x, percent_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_image

    def show_live_classification_screen(self):
        """Display the live classification screen"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
            
        self.clear_screen()
        self.current_screen = "live_classification"
        self.root.configure(bg="#1a1a1a")
        
        # Set a larger window for live classification but not full screen
        self.root.geometry("1000x800")
        
        # Main container using pack with proper expansion control
        main_frame = tk.Frame(self.root, bg="#1a1a1a")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title - Row 0
        title_frame = tk.Frame(main_frame, bg="#1a1a1a")
        title_frame.pack(fill="x", pady=(0, 10))
        
        title_label = tk.Label(title_frame, text="Live Classification", 
                              font=("Arial", 20, "bold"), fg="white", bg="#1a1a1a")
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Real-time Classification with Enhanced Controls", 
                                 font=("Arial", 12), fg="#bdc3c7", bg="#1a1a1a")
        subtitle_label.pack()
        
        # Settings panel - Row 1
        settings_frame = tk.LabelFrame(main_frame, text="Classification Settings", 
                                      font=("Arial", 11, "bold"), fg="white", bg="#2c3e50",
                                      padx=10, pady=8)
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Row 1: Confidence and Class filter
        row1_frame = tk.Frame(settings_frame, bg="#2c3e50")
        row1_frame.pack(fill="x", pady=2)
        
        # Confidence threshold
        tk.Label(row1_frame, text="Confidence:", font=("Arial", 9, "bold"), 
                fg="white", bg="#2c3e50").pack(side="left", padx=(0, 5))
        
        self.live_confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        live_conf_scale = tk.Scale(row1_frame, from_=0.1, to=0.9, resolution=0.05,
                                  orient="horizontal", variable=self.live_confidence_var,
                                  command=self.update_confidence_threshold,
                                  length=150, showvalue=True,
                                  bg="#2c3e50", fg="white", highlightbackground="#2c3e50")
        live_conf_scale.pack(side="left", padx=(0, 20))
        
        # Class filter
        tk.Label(row1_frame, text="Class Filter:", font=("Arial", 9, "bold"),
                fg="white", bg="#2c3e50").pack(side="left", padx=(0, 5))
        
        # Create a frame for class checkboxes with fixed height
        class_check_frame = tk.Frame(row1_frame, bg="#2c3e50", height=30)
        class_check_frame.pack(side="left", fill="x", expand=True)
        class_check_frame.pack_propagate(False)
        
        # Create canvas for class checkboxes
        class_canvas = tk.Canvas(class_check_frame, bg="#2c3e50", highlightthickness=0, height=30)
        class_scrollbar = ttk.Scrollbar(class_check_frame, orient="horizontal", command=class_canvas.xview)
        class_scrollable = tk.Frame(class_canvas, bg="#2c3e50")
        
        class_scrollable.bind(
            "<Configure>",
            lambda e: class_canvas.configure(scrollregion=class_canvas.bbox("all"))
        )
        
        class_canvas.create_window((0, 0), window=class_scrollable, anchor="nw")
        class_canvas.configure(xscrollcommand=class_scrollbar.set)
        
        # Create class checkboxes
        self.live_class_vars = {}
        for class_name in self.class_names.values():
            var = tk.BooleanVar(value=self.selected_classes[class_name])
            self.live_class_vars[class_name] = var
            
            cb = tk.Checkbutton(class_scrollable, text=class_name, variable=var,
                               command=self.update_live_class_filter,
                               bg="#2c3e50", fg="white", selectcolor="#34495e",
                               activebackground="#2c3e50", activeforeground="white")
            cb.pack(side="left", padx=3)
        
        class_canvas.pack(side="top", fill="x")
        class_scrollbar.pack(side="bottom", fill="x")
        
        # Video display - Fixed size container
        video_container = tk.Frame(main_frame, bg="#3498db", padx=2, pady=2)
        video_container.pack(fill="both", expand=True, pady=(0, 10))
        
        # Set a fixed maximum size for the video display
        self.video_frame = tk.Frame(video_container, bg="black", width=800, height=600)
        self.video_frame.pack(expand=True)
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill="both", expand=True)
        
        # Controls - Fixed at bottom
        controls_frame = tk.Frame(main_frame, bg="#2c3e50", padx=15, pady=10)
        controls_frame.pack(fill="x", pady=(0, 5))
        
        controls_title = tk.Label(controls_frame, text="Controls", 
                                 font=("Arial", 12, "bold"), fg="white", bg="#2c3e50")
        controls_title.pack(pady=(0, 8))
        
        controls_buttons = tk.Frame(controls_frame, bg="#2c3e50")
        controls_buttons.pack()
        
        # Control buttons
        self.record_btn = ModernButton(controls_buttons, "● Stop Recording", 
                                     command=self.toggle_recording,
                                     width=160, height=45, color="#e74c3c", 
                                     hover_color="#c0392b")
        self.record_btn.pack(side="left", padx=10)
        
        back_btn = ModernButton(controls_buttons, "Back to Main", 
                              command=self.stop_live_classification,
                              width=140, height=45, color="#95a5a6", hover_color="#7f8c8d")
        back_btn.pack(side="left", padx=10)
        
        stats_btn = ModernButton(controls_buttons, "View Stats", 
                               command=self.show_statistics_dashboard,
                               width=120, height=45, color="#3498db", hover_color="#2980b9")
        stats_btn.pack(side="left", padx=10)
        
        quit_btn = ModernButton(controls_buttons, "Quit", 
                              command=self.on_closing,
                              width=100, height=45, color="#e74c3c", hover_color="#c0392b")
        quit_btn.pack(side="left", padx=10)
        
        # Status
        status_frame = tk.Frame(main_frame, bg="#1a1a1a", pady=5)
        status_frame.pack(fill="x")
        
        self.live_status_label = tk.Label(status_frame, text="Starting webcam...", 
                                         font=("Arial", 11, "bold"), fg="#2ecc71", 
                                         bg="#1a1a1a")
        self.live_status_label.pack()
        
        # Initialize live classifier
        self.live_classifier = LiveClassifierGUI(self.model, self, self.stats)
        self.root.after(100, self.start_live_classification)
    
    def update_live_class_filter(self):
        """Update class filter for live classification"""
        for class_name, var in self.live_class_vars.items():
            self.selected_classes[class_name] = var.get()
    
    def start_live_classification(self):
        """Start the live classification"""
        if hasattr(self, 'live_classifier'):
            self.live_classifier.start()
    
    def toggle_recording(self):
        """Toggle recording state"""
        if hasattr(self, 'live_classifier'):
            self.live_classifier.toggle_recording()
    
    def stop_live_classification(self):
        """Stop live classification and return to main menu"""
        if hasattr(self, 'live_classifier'):
            self.live_classifier.stop()
            del self.live_classifier
        # Reset window size when returning to main menu
        self.root.geometry("900x700")
        self.show_main_menu()
    
    def update_video_frame(self, frame):
        """Update the video display with new frame"""
        if self.current_screen != "live_classification":
            return
            
        # Convert frame to ImageTk format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Get the fixed size of our video frame
        display_width = 800
        display_height = 600
        
        # Resize image to fit our fixed display size while maintaining aspect ratio
        img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def update_status(self, text, color="#2ecc71"):
        """Update the status label"""
        if hasattr(self, 'live_status_label'):
            self.live_status_label.config(text=text, fg=color)
    
    def update_record_button(self, recording):
        """Update the record button text and color"""
        if hasattr(self, 'record_btn'):
            if recording:
                text = "● Stop Recording"
                color = "#e74c3c"  # Red
            else:
                text = "▶ Start Recording" 
                color = "#27ae60"  # Green
            
            self.record_btn.update_text(text)
            self.record_btn.update_color(color)
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = MainApplication()
    app.run()