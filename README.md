# YOLOv8 Object Classification and Detection Repository

This repository showcases two full deep‑learning pipelines built using **YOLOv8**:

- **Object Classification System** — A full Tkinter-based GUI application supporting live webcam classification, batch image classification, class filtering, confidence control, PDF reporting, and auto video recording.
- **Object Detection System** — A YOLOv8 detection interface with live detection, image folder detection, filtering, result visualization, and statistics.

Both systems come with trained model weights, result logs, and graphical user interfaces.

---
## 📸 Interface Screenshots
Below are sample screenshots of the interface in action:

- **Classification Main Menu:**

<img width="563" height="458" alt="image" src="https://github.com/user-attachments/assets/8cf61df2-8219-4aab-8b3d-8bd41137a50e" />

---
- **Batch Image Classification Screen:**

<img width="563" height="458" alt="image" src="https://github.com/user-attachments/assets/24869aa8-8454-4685-90ad-48c3fca422a7" />

---
- **Live Webcam Classification:**

<img width="563" height="458" alt="image" src="https://github.com/user-attachments/assets/d7203828-1c9f-4de0-8e81-e13038b114b0" />

---

## 📁 Repository Structure

```
Object_Classification/
├── classification_results/
├── runs/classify/train/weights/
│   ├── best.pt
│   └── last.pt
└── Classification_Interface.py

Object_Detection/
├── classification_results/
├── runs/detect/train_session_final/weights/
│   ├── best.pt
│   └── last.pt
└── Detection_Interface.py

PDE3802_Assessment_Part_A_File.pdf

UI_User_Guide.pdf
```

---

## ✅ Features

### **Object Classification System (GUI Application)**
- Full Tkinter application with multiple screens
- Live classification with FPS, processing time, class filtering, confidence slider, auto‑recording
- Batch image classification with overlays and folder selection
- Statistics dashboard with scrollable class list
- Automated PDF report generation

### **Object Detection System**
- YOLOv8 detection interface for webcam and folder
- Adjustable confidence and class filtering
- Recording of annotated video
- Lightweight interface for quick testing

---

## 🛠️ Installation Instructions (Windows Summary)

### 1. Clone the repository
```bash
git clone https://github.com/5AMUELBASH/AI_IN_ROBOTICS.git
cd AI_IN_ROBOTICS
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the applications
Make reference to the file "UI_User_Guide.pdf" for instructions on using the classification and detection interfaces.

---

## ✅ System Requirements 
Use these for reference or individual installations. Otherwise, install with the requirements text file.

```
python==3.12.9
ultralytics==8.1.0
opencv-python==4.10.0.84
pillow==10.2.0
matplotlib==3.8.3
numpy==1.26.4
pandas==2.2.1
python-dateutil==2.9.0
tk==0.1.0
scipy==1.12.0
PyYAML==6.0.1
```

---

## 📌 Notes
- Modify `model_path` in scripts if model files are moved
- PDF reports auto‑save when the GUI closes
- Statistics of current session available mid session or upon system completion

---

## 📄 License
Add your license text here.