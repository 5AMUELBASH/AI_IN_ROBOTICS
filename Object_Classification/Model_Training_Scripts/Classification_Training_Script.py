from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="Object_Classification_Dataset",  
        epochs=50,
        imgsz=320, 
        batch=32,          
        workers=8,          # for parallel data loading
        device=0,          
        amp=True,           # automatic mixed precision (Tensor Core boost)
        cache=True,         # caches images in RAM for faster epochs
        patience=10,      
        dropout=0.2        # improves generalization
    )
