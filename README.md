
# 🌊 OceanGuard-AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-Educational-green)

An AI-powered underwater monitoring system that detects debris and algae to assess water quality and pollution levels in real-time.

## 📖 Project Overview

Water pollution and underwater debris are major environmental concerns affecting marine ecosystems and human health. Manual inspection is slow and resource-intensive. **OceanGuard-AI** provides an automated, image-based solution using **YOLOv8** to detect underwater debris and algae, delivering instant ecological insights and actionable water quality metrics.

## ✨ Key Features

- **Automated Object Detection:** Identifies plastic waste (bottles, bags, nets) and algae species.
- **Pollution Severity Scoring:** Calculates a pollution index classifying water as Low, Moderate, High, or Critical.
- **Interactive Web Interface:** A Streamlit-based dashboard for easy image uploading and analysis.
- **Visual Analytics:** Outputs annotated images with bounding boxes, confidence scores, and analytical tables summarizing object counts and impact.
- **Geospatial Tracking:** Includes an interactive map to visualize monitored sites and pollution hotspots.

## 🛠️ Tech Stack

- **Machine Learning:** YOLOv8 (Ultralytics) for Convolutional Neural Network (CNN) based object detection.
- **Frontend / UI:** Streamlit
- **Language:** Python

## 🚀 Getting Started

Follow these steps to run OceanGuard-AI on your local machine.

### Prerequisites
Make sure you have Python installed. It is recommended to use a virtual environment.

### Installation
1. Clone the repository:
   ```bash
     git clone https://github.com/nowsheen19jahan/OceanGuard-AI.git
      cd OceanGuard-AI

    ```
  
2. Install the required dependencies:
  ```bash
    pip install -r requirements.txt
  
  ```



### Running the App

Start the Streamlit application by running:

```bash
     streamlit run app.py

```

*The app will automatically open in your default web browser.*

## 📈 Future Enhancements

* Improve algae detection accuracy and coverage across diverse underwater environments.
* Integrate multiple sample sites for aggregated, large-scale pollution reporting.
* Enhance visualization of pollution hotspots on interactive maps.

## 🎓 About the Project

This project was developed as part of the **Shell–Edunet Skills4Future AICTE Internship**, organized by **Edunet Foundation** in collaboration with **AICTE** and **Shell**. The internship focuses on building *Green Skills through Artificial Intelligence (AI)*, empowering students to create innovative and sustainable solutions for real-world environmental challenges.

## 📜 License

© 2025 [OceanGuard AI](https://github.com/nowsheen19jahan/OceanGuard-AI) • For educational and research purposes.

