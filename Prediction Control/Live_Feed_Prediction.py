from ultralytics import YOLO
import cv2

model_path = "runs/classify/corn_classifier3/weights/best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to access the camera.")
    exit()

print("🎥 Press 'q' to quit live classification.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(rgb_frame, imgsz=224, verbose=False, device='cpu')

    top_result = results[0].probs
    class_id = top_result.top1
    class_name = model.names[class_id]
    confidence = top_result.top1conf.item()

    label = f"{class_name} {confidence:.1%}"
    cv2.putText(frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

    cv2.imshow("Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
