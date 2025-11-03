from ultralytics import YOLO
from pathlib import Path

ROOT    = Path("/home/omar.elalaily/Mapillary")
WEIGHT  =  ROOT / "output_runs_modified" / "modified_train" / "weights" / "best.pt"
RUNS    = ROOT / "output_runs_modified"
SOURCE  = ROOT / "dataset" / "images" / "test"

# Load the best weights after training
model = YOLO(WEIGHT)

# Run inference on test images (no labels needed)
model.predict(
    source=SOURCE,
    imgsz=1024,
    batch=8,
    # conf=0.25,         # optional: confidence threshold
    device=0,
    save=True,
    save_txt=True,     
    project=RUNS,
    name='modified_test_predictions'
)
