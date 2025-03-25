# Face Spoofing Detection Using CNN and Patch CNN Models

## Project Overview

In this project, we thoroughly explored the critical need for **face spoofing detection**, highlighting the increasing prevalence of spoofing techniques and the vulnerabilities they expose in authentication systems. Spoofing attacks have become a significant concern, especially in security systems that rely on facial recognition for user authentication. 

We discussed various detection techniques, including:
- **Motion-based techniques**
- **Multi-spectral based techniques**
- **Texture-based techniques**

However, we identified key limitations in these methods, which made them struggle to effectively differentiate between genuine and spoofed faces.

## Objective

To address this challenge, we developed a **Patch Based Convolutional Neural Network (CNN)** model using the **CelebA Spoof dataset**. This dataset provided a robust framework for building and training our model, which consisted of 100,000 images (50,000 real and 50,000 spoofed faces). The goal was to accurately classify live and spoofed faces, achieving a high accuracy rate.

## Approach

### CNN Model
- We built and trained a CNN model on the CelebA Spoof dataset.
- The model demonstrated impressive performance, achieving an accuracy of **91.76%**, indicating its strong ability to accurately classify live and spoofed faces.

### Patch CNN Model
To further enhance the accuracy, we developed a **Patch CNN model** that focuses on analyzing local regions of the input image. This approach allowed the model to detect fine-grained differences between real and spoofed faces, leading to better performance. The Patch CNN model achieved an impressive accuracy of **93.75%** on the CelebA Spoof dataset.

### Integration for Real-Time System
We outlined a flowchart for integrating the model into a real-time system, showcasing its potential practical application in real-world scenarios.

### Evaluation on NUAA Impostor Dataset
We also tested both CNN and Patch CNN models on the **NUAA Impostor dataset** to assess their generalizability:
- The **CNN model** achieved an accuracy of **67.56%**.
- The **Patch CNN model** outperformed the CNN model with an accuracy of **75.81%**.

## Results & Conclusion

- **Patch CNN** outperformed the standard **CNN** in both real-time detection and dataset evaluation, demonstrating superior generalization and accuracy.
- Its ability to focus on local regions of the input image proved crucial in effectively distinguishing between genuine and spoofed faces.
- These results highlight the potential of **Patch CNN** as a reliable and robust solution for **face spoofing detection** in authentication systems.

## Future Work

- Explore additional datasets for further validation and testing.
- Investigate different architectures and techniques, such as **Attention Mechanisms** or **Transfer Learning**, to improve performance.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV

