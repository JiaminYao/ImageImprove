# 🐶 Dog Image Grooming Salon  
**Digital Image Enhancement for Pet Photography**

## 📌 Project Overview

Dog photography is one of the most popular forms of visual content across social media, pet services, veterinary records, and consumer products. However, capturing high-quality dog images is challenging due to **uneven lighting, motion blur, low sharpness, and distracting backgrounds**.

This project presents a **three-stage digital image enhancement pipeline** designed to transform raw dog photos into visually appealing, professional-quality images. The system integrates **localized contrast enhancement**, **edge-preserving sharpness restoration**, and **automated background suppression**, with performance evaluated using both **quantitative metrics** and **qualitative visual analysis** :contentReference[oaicite:0]{index=0}.


## 📊 Dataset Description

To ensure diversity in lighting, resolution, and background complexity, images were selected from three public datasets:

| Dataset | Images Used | Year | Purpose |
|------|-----------|------|--------|
| Stanford Dogs Dataset | 15 | 2011 | Contrast & background experiments |
| Tsinghua Dogs Dataset | 10 | 2020 | Noise reduction & sharpening |
| Exclusively-Dark-Image Dataset | 5 | 2015 | Low-light enhancement |

- **Total Images:** 30  
- **Subjects:** Single dog per image  
- **Conditions Covered:**  
  - Dark lighting  
  - Overexposure  
  - Motion blur  
  - Low resolution  
  - Low-clutter and high-clutter backgrounds  


## 🧠 Methodology

The enhancement pipeline consists of **three sequential stages**, each targeting a key limitation of dog photography.

### 1️⃣ Localized Contrast Enhancement
To address uneven or extreme lighting:
- Global Histogram Equalization (GHE)
- Adaptive Histogram Equalization (AHE)
- **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**

CLAHE prevents noise over-amplification while improving fur texture and facial details.


### 2️⃣ Edge-Preserving Noise Reduction & Sharpness Enhancement
To restore fine details lost to blur or noise:
- Gaussian Smoothing
- Median Filtering
- **Unsharp Masking**

Unsharp masking combines smoothing with high-frequency detail amplification to enhance edges without introducing artifacts.


### 3️⃣ Automated Background Suppression
To reduce visual distractions while preserving the subject:
- Luminance-Guided Blur
- Gaussian Blur with Foreground Masking
- **Distance-Based Selective Blur**

Distance-based selective blur produces natural depth-of-field effects similar to professional portrait photography.


## 📈 Evaluation Metrics

Each experiment is evaluated using a combination of **objective image quality metrics** and **subjective visual assessment**.

### Quantitative Metrics
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Mean Squared Error (MSE)
- Entropy
- Edge Strength (Sobel & Laplacian Variance)
- Background Entropy
- Foreground-Background Contrast

### Evaluation Strategy
- Metrics are normalized to a common scale
- Composite scores computed using weighted sums
- Best parameters selected per method
- Final results assessed qualitatively through side-by-side comparisons


## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries & Tools:**
  - OpenCV
  - NumPy
  - SciPy
  - Matplotlib
  - scikit-image  


## 🎯 Conclusion

This project demonstrates that **adaptive, multi-stage image enhancement** significantly improves the visual quality of dog photographs under real-world conditions.

Key findings:
- **CLAHE** offers the best balance between contrast enhancement and realism
- **Unsharp Masking** effectively restores edge clarity without excessive artifacts
- **Distance-Based Selective Blur** provides the most natural and visually appealing background suppression

The proposed pipeline is efficient, visually robust, and suitable for **social media, veterinary documentation, e-commerce, and mobile applications**, highlighting the value of classical image processing techniques for real-world visual enhancement tasks.
