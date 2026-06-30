# Short Note

## Spot the Fake Photo – Approach

The objective of this assignment was to determine whether an input image is a genuine photograph or a photograph of a digital screen. While the two classes often appear visually similar, screen recaptures typically contain subtle artifacts introduced by display hardware and the camera imaging pipeline, such as moiré patterns, pixel grid structures, reflections, texture distortions, and frequency-domain inconsistencies.

Rather than relying on a single approach, I treated this as a model selection problem and evaluated both classical machine learning and deep learning techniques.

I first collected a balanced custom dataset consisting of **102 smartphone-captured images (51 real photographs and 51 screen recaptures)** under different lighting conditions, viewing angles, backgrounds, and object categories. After performing exploratory data analysis, I extracted several handcrafted image descriptors—including Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), Fast Fourier Transform (FFT), Laplacian variance, edge density, reflection statistics, and color features—to establish classical machine learning baselines. Using these features, I trained both Random Forest and XGBoost classifiers.

Although these models produced encouraging results, I expected transfer learning to capture more complex visual patterns than handcrafted features alone. Consequently, I evaluated multiple ImageNet-pretrained convolutional neural networks, including **MobileNetV3 Small, EfficientNet-B0, ResNet18, and ResNet34**, using **Stratified 5-Fold Cross Validation**. Every model was trained under the same experimental setup using transfer learning, AdamW optimization, learning-rate scheduling, data augmentation, and early stopping to ensure a fair comparison.

Among all evaluated architectures, **ResNet34** achieved the strongest overall performance with a **mean cross-validation accuracy of 85.33%**, a **mean F1-score of 86.06%**, and a **best validation fold accuracy of 95.00%**. While EfficientNet-B0 achieved a perfect score on one validation fold, its average performance was lower and less consistent across folds. Since the dataset is relatively small, I prioritized consistency across validation splits rather than selecting a model based on a single high-performing fold. Based on this evaluation, ResNet34 was selected as the final deployment model and subsequently retrained using the complete dataset to maximize the available training data before deployment.

The final solution consists of a lightweight **PyTorch inference pipeline** (`predict.py`) together with an interactive **Streamlit application** for real-time demonstration. Inference is performed entirely **on-device**, achieving an average latency of approximately **30–40 ms per image** on an Apple MacBook Air M2 CPU. Because no external APIs or cloud infrastructure are required, the effective inference cost is **approximately $0 per image**, making the solution suitable for lightweight real-time deployment.

## What I Would Improve

The primary limitation of this project is the size and diversity of the dataset. Given additional time, I would collect substantially more images across different smartphone cameras, display technologies (LCD, OLED, AMOLED, Mini-LED), lighting conditions, viewing angles, and exposure settings to improve generalization.

I would also investigate frequency-domain feature fusion, Vision Transformer architectures, model quantization for mobile deployment, and threshold optimization using ROC and Precision–Recall analysis. In a production environment, I would continuously collect newly observed fraud examples, perform hard-negative mining, and periodically retrain the model to ensure the detector remains effective as fraud patterns evolve.