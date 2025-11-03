if __name__ == '__main__':
    from ultralytics import YOLO
    from pathlib import Path

    ROOT    = Path("/home/omar.elalaily/Mapillary")        # server base
    MY_YAML = ROOT / "weights"    / "mapillary_modified.yaml"
    WEIGHTS = ROOT / "weights"   / "yolov8.yaml"
    RUNS    = ROOT / "output_runs_modified"

    model = YOLO(WEIGHTS)

    #Train on my custom dataset
    model.train(
                data=MY_YAML,
                epochs=100,
                imgsz=1024,
                batch=8,
                device=[0,1],
                lr0=1e-4,
                lrf=1e-5,
                cos_lr=True,        
                patience=20,
                project=RUNS,
                name="modified_train")
    #Evaluate trained model
    metrics = model.val(
                        data=MY_YAML,
                        split='val',
                        device=[0,1],
                        project=RUNS,
                        name='modified_val',
                        verbose=True,
                        plots=True)
    print(metrics)

    WEIGHT =  ROOT / "output_runs_modified" / "modified_train" / "weights" / "best.pt"
    RUNS_fine    = ROOT / "fine_tune_output_modified"

    model = YOLO(WEIGHT, task='detect')

    model.train(
    data=MY_YAML,
    epochs=20,      
    batch=8,
    imgsz=1024,
    lr0=1.3e-4,       
    lrf=1.3e-5,
    device=[0,1],
    cos_lr=True,
    patience=10,
    project=RUNS_fine,
    name="modified_train_fine_tune")

    metrics_fine_tune = model.val(
                        data=MY_YAML,
                        split='val',
                        device=[0,1],
                        project=RUNS,
                        name='modified_val_fine_tune',
                        verbose=True,
                        plots=True)
    print(metrics_fine_tune)
