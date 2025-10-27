from ultralytics import YOLO
import os
import cv2
import numpy as np

model_path = "runs/classify/corn_classifier3/weights/best.pt"

# Folder with input images
input_folder = "Test_Images"

# Folder to save annotated images
output_folder = "Test_Results"
os.makedirs(output_folder, exist_ok=True)

# Load classification model
model = YOLO(model_path)

# Loop through images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)

        results = model.predict(source=img_path, imgsz=224, verbose=False)
        top_result = results[0].probs

        class_id = top_result.top1
        class_name = model.names[class_id]
        confidence = top_result.top1conf.item()

        # Load with OpenCV for drawing
        img = cv2.imread(img_path)

        # Create text label
        label = f"{class_name} {confidence:.1%}"

        # Draw text on image
        cv2.putText(img,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        # Display result
        cv2.imshow("Prediction", img)

        # Wait for key press before continuing
        key = cv2.waitKey(0)
        if key == ord('q'):  # optional early exit if you hit q
            break

        # Save output
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)

        print(f"✅ {filename} → {label}")

cv2.destroyAllWindows()
print("🎉 Classification display complete. Annotated images saved to:", output_folder)
