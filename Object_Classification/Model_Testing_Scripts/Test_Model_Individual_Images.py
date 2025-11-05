from ultralytics import YOLO
import cv2
import os
import glob

# Load your trained YOLOv8 classification model
model = YOLO('runs_Tunmise/classify/yolov8n_cls_V3/weights/best.pt')
# Path to folder containing test images
test_folder = 'Test_Images'

# Create output directory if it doesn't exist
output_folder = 'Prediction_Results'
os.makedirs(output_folder, exist_ok=True)

# Get all image files (common extensions)
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(test_folder, ext)))

print(f"Found {len(image_paths)} images to process...")

# Process each image
for image_path in image_paths:
    # Perform prediction
    results = model(image_path)
    
    # Extract results
    result = results[0]
    top1 = result.probs.top1
    top1_conf = result.probs.top1conf
    class_names = model.names
    
    # Read image
    img = cv2.imread(image_path)
    
    # Get image dimensions to scale text appropriately
    height, width = img.shape[:2]
    
    # Calculate font scale based on image size
    font_scale = max(1.5, min(3.0, width / 600))  # Adjust scale based on image width
    
    # Calculate thickness based on font scale
    thickness = max(2, int(font_scale))
    
    # Prepare text
    label = f"{class_names[top1]}: {top1_conf:.2f}"
    
    # Get text size to position it properly
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    
    # Calculate text position (top-left with some padding)
    text_x = 20
    text_y = text_size[1] + 30
    
    # Add a background rectangle for better text visibility
    cv2.rectangle(img, 
                  (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), 
                  (0, 0, 0), -1)  # Black background
    
    # Add the text in white for high contrast
    cv2.putText(img, label, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Generate output path
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    
    # Save annotated image
    success = cv2.imwrite(output_path, img)
    
    if success:
        print(f"Processed: {filename} -> {class_names[top1]} ({top1_conf:.2f})")
    else:
        print(f"Failed to save: {filename}")

print(f"\nAll images saved to '{output_folder}' folder!")