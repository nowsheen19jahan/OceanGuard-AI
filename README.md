# OceanGuard-AI
AI-powered water quality monitoring system that classifies water conditions and pollution levels using image-based CNN and YOLO object detection models.

---

## Internship Information
This project is developed as part of the **Shell–Edunet Skills4Future AICTE Internship**, organized by **Edunet Foundation** in collaboration with **AICTE** and **Shell**.

The internship focuses on building **Green Skills through Artificial Intelligence (AI)**, empowering students to create innovative and sustainable solutions for real-world environmental challenges.

---

## Project Title
**OceanGuardAI – AI-Powered Water Quality Monitoring System**

---

## Problem Statement
Water pollution is a major environmental concern affecting marine ecosystems, human health, and biodiversity.  
Recent cases like the **brain-eating amoeba infection (Naegleria fowleri)** in Kerala highlight the urgent need for **early detection of unsafe water** conditions.

Most detection methods rely on **chemical lab testing** or **manual sampling**, which are slow and not scalable.  
There is a lack of **AI-driven, image-based systems** that provide real-time insights into water quality from simple images captured via smartphones or drones.

---

## Project Proposal
The goal is to develop an **AI-based system** using **CNNs** for classification and **YOLOv8** for object detection to automatically detect and classify:

- Clean Water
- Slightly Polluted Water
- Heavily Polluted Water
- Plastic Waste Present
- Hazardous Condition

The system will eventually be integrated into a **Streamlit web app**, allowing users to:

- Upload water images
- Get an instant prediction with a **cleanliness/pollution score**
- Receive actionable advice like “Safe for Use” or “Avoid Contact”
- (Future) Visualize pollution zones on an interactive map

---

## Week 1 Progress
### ✅ Completed
1. **Dataset Collection & Exploration**
   - Collected water pollution and debris datasets from Kaggle.
   - Explored datasets for structure, missing labels, and image quality.

2. **Data Preprocessing**
   - Converted datasets to **YOLO format** (train/test/val splits with images and labels).
   - Flattened folder structures and ensured matching images & labels.
   - Created **data YAML files** for YOLO training.

3. **Repository Preparation**
   - Organized folders for datasets, notebooks, and model storage.
   - Documented folder structure and dataset details.

---

## Week 2 Progress
### ✅ Completed
1. **YOLO Training – Underwater Plastics**
   - Trained **YOLOv8n** model on underwater plastics dataset.
   - Saved trained weights (`best.pt` and `last.pt`).
   - Validated model on test images and obtained metrics.

2. **Saved Model & Notebook**
   - Stored trained model and notebook locally for submission.
   - Prepared folder structure for GitHub.

3. **Next Steps**
   - Train YOLO models for:
     - **Algae Detection**
     - **SeaClear Marine Debris**
   - Integrate all models into a **Streamlit web app** for real-time predictions.

---

## Dataset Information
The project uses **freely available Kaggle datasets**:

1. **Water Pollution Dataset** – Kaggle  
   🔗 [Link](https://www.kaggle.com/datasets/adityapandey360/water-pollution-dataset)

2. **Plastic Waste in Water Dataset** – Kaggle  
   🔗 [Link](https://www.kaggle.com/datasets/techsash/waste-classification-data)

3. **Clean vs Polluted Water Dataset** – Kaggle  
   🔗 [Link](https://www.kaggle.com/datasets/iamsouravbanerjee/water-quality-dataset)

---

