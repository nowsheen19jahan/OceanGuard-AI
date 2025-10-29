# OceanGuard-AI
AI-powered water quality monitoring system that classifies water conditions and pollution levels using image-based CNN models.
## Internship Information  

This project is developed as part of the **Shell–Edunet Skills4Future AICTE Internship** organized by **Edunet Foundation** in collaboration with **AICTE** and **Shell**.  

The internship focuses on building **Green Skills through Artificial Intelligence (AI)**, empowering students to create innovative and sustainable solutions for real-world environmental challenges.  

---

## Project Title  
**OceanGuardAI – AI-Powered Water Quality Monitoring System**

---

##  Problem Statement  

Water pollution continues to be one of the biggest environmental concerns worldwide. Contaminated water sources affect marine ecosystems, human health, and biodiversity.  
Recently, cases like the **brain-eating amoeba infection (Naegleria fowleri)** reported in Kerala have drawn attention to the **urgent need for early detection of unsafe water** conditions.  

Most existing detection methods rely on **chemical lab testing** or **manual sampling**, which are slow and not scalable.  
There is a lack of **AI-driven, image-based detection systems** that can provide real-time insights into the **condition of water bodies** from simple images captured via smartphones or drones.  

---

##  Project Proposal  

The goal of this project is to develop an **AI-based image classification model** using **Convolutional Neural Networks (CNNs)** that can automatically detect and classify **water quality conditions** from images.  

The system will classify images into five categories:  
1. Clean Water  
2. Slightly Polluted Water  
3. Heavily Polluted Water  
4. Plastic Waste Present  
5. Hazardous Condition  

The model will be integrated into a **Streamlit web application**, where users can:  
- Upload water images  
- Get an instant prediction with a **cleanliness score (0–100%)**  
- Receive **actionable advice** such as “Safe for Use” or “Avoid Contact”  
- (Future Scope) Visualize **pollution zones on an interactive map**

---

##  Dataset Information  

The project uses **freely available and easy-to-access datasets** from **Kaggle**.

###  Datasets Used  
1. **Water Pollution Dataset** – Kaggle  
   🔗 [https://www.kaggle.com/datasets/adityapandey360/water-pollution-dataset](https://www.kaggle.com/datasets/adityapandey360/water-pollution-dataset)  
   → Contains labeled images of clean and polluted water surfaces.  

2. **Plastic Waste in Water Dataset** – Kaggle  
   🔗 [https://www.kaggle.com/datasets/techsash/waste-classification-data](https://www.kaggle.com/datasets/techsash/waste-classification-data)  
   → Includes water bodies containing visible plastic, metal, and waste materials.  

3. **Clean vs Polluted Water Dataset** – Kaggle  
   🔗 [https://www.kaggle.com/datasets/iamsouravbanerjee/water-quality-dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/water-quality-dataset)  
   → General-purpose dataset for visual classification of clean vs dirty water.  

---

