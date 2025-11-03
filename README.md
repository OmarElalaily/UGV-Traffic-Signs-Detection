# UGV Autonomous Traffic Sign Detection (YOLOv8)
**Specialized Deep Learning Model** for real-time identification and classification of **American-Style Traffic Signs** to facilitate autonomous navigation for Unmanned Ground Vehicles (UGVs).

---

### Project Objectives

* **Goal:** To deliver a **high-accuracy** and **robust** Computer Vision pipeline capable of reliably classifying all necessary US road signs.
* **Function:** Implement a model trained exclusively on a subset of the Mapillary Vistas Traffic Sign dataset to achieve optimal performance specific to the **226 relevant US sign classes**.

### Methodology Highlights

* **Architecture:** Utilizes the **YOLOv8** object detection framework.
* **Unique Data Strategy: Class Re-Mapping**
    * The model was initially trained using the massive **Mapillary Vistas Traffic Sign dataset (401 classes)**.
    * The data was then processed, and classes were **remapped and filtered down to 226 specific classes** required for autonomous operation in the United States, optimizing the model's focus and reducing complexity.

### Dataset Source

The core data for this project originates from the official Mapillary Vistas dataset.

* **Dataset Link:** [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)
    * *Note: This repository contains the code and configuration for using the **filtered 226-class version** of the dataset.*


### Inference Results

Below are samples of the model running on unseen validation data:

**1. Max Speed Sign Detection**

![Detection of the Max Speed Sign](./images/speed_limit.jpg)
![Detection of the Max Speed Sign](./images/speed_limit_2.jpg)

**2. No Entry Sign Detection**

![Detection of the No Entry Sign](./images/do_not_enter.jpg)
![Detection of the No Entry Sign](./images/no-enter.jpg)

**3. Stop Sign Detection**

![Detection of the Stop Sign](./images/stop.jpg)
![Detection of the Stop Sign](./images/stop2.jpg)
![Detection of the Stop Sign](./images/stop-donot.jpg)


<hr>
