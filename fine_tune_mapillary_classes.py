if __name__ == '__main__':
    from ultralytics import YOLO
    from pathlib import Path

    ROOT    = Path("/home/omar.elalaily/Mapillary")
    MY_YAML = ROOT / "weights"    / "mapillary_modified.yaml"
    WEIGHT =  ROOT / "output_runs_modified" / "modified_train2" / "weights" / "best.pt"
    RUNS    = ROOT / "fine_tune_output_modified"

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
    project=RUNS,
    name="modified_train_fine_tune")

    metrics = model.val(
                        data=MY_YAML,
                        split='val',
                        device=[0,1],
                        project=RUNS,
                        name='modified_val_fine_tune',
                        verbose=True,
                        plots=True)
    print(metrics)