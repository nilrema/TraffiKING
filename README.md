# Traffic Sign Recognition Using Convolutional Neural Networks (CNN) - TraffiKING Project

## Description

This project explores the use of Convolutional Neural Networks (CNNs) for the classification of traffic signs. It focuses on developing, training, and evaluating a custom-designed CNN model named "TraffiKING". The project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Additionally, the performance of the custom TraffiKING model is compared against state-of-the-art networks: ResNet-18, EfficientNet-B0, MobileNetV2, and DenseNet121. The study highlights the potential of CNNs for robust traffic sign recognition and provides insights into optimizing neural network architectures for this task.

## Dataset

* **Source:** The German Traffic Sign Recognition Benchmark (GTSRB) dataset. Available on Kaggle: [GTSRB Dataset Link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
* **Content:** Contains over 50,000 labeled images across 43 different traffic sign classes. The images reflect real-world challenges like varying illumination, rotations, partial occlusions, perspective distortions, blurring, and varying resolutions.
* **Preprocessing:**
    * Images were resized to $64\times64$ pixels.
    * Pixel values were normalized to the range [0, 1] after conversion to RGB format.
    * The dataset was split into training (80%) and test (20%) sets. A validation set was also used during training.

## Model Architecture: TraffiKING (Custom CNN)

The custom "TraffiKING" CNN model consists of:
1.  **Input Layer:** Takes $64\times64$ RGB images (3 channels).
2.  **Feature Extraction:**
    * Initial `BatchNorm2d` + `ReLU` activation.
    * `Conv1`: 32 filters ($3\times3$), stride 1, padding 1.
    * `BatchNorm2d` + `ReLU` activation.
    * `Conv2`: 64 filters ($3\times3$), stride 1, padding 1.
    * `MaxPool2d`: $2\times2$ pooling.
3.  **Flattening Layer:** Flattens feature maps.
4.  **Fully Connected Layer (FC1):** Connects flattened features to 43 output neurons (one per class).
5.  **Output Layer:** Outputs a vector of size 43 representing class probabilities.

*(See Figure 2 in the original document for a visual representation)*

## Experimental Process

* **Environment:** Python (Jupyter Notebook).
* **Libraries:** NumPy, OpenCV, OS, Torchvision (PyTorch).
* **Training:**
    * **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
    * **Optimizer:** Adam (`optim.Adam`) with an initial learning rate of 0.001.
    * **Scheduler:** Step Learning Rate Scheduler (`lr_scheduler.StepLR`), reducing LR by 0.1 every 4 epochs.
    * GPU acceleration was used.
    * The best model based on validation accuracy was saved.
* **Evaluation:** The model was evaluated on the test set using accuracy as the metric. Comparisons were made with both pretrained and randomly initialized versions of ResNet-18, EfficientNet-B0, DenseNet121, and MobileNetV2. Domain change analysis was also performed.

## Results

* **TraffiKING (Custom Model):**
    * Validation Accuracy (Pretrained/Initialized): 97.28%.
    * Test Accuracy (Domain Change): 64.32%.
    * State Dict Size: 10.8MB.
    * Parameters: ~2.84M.
* **Comparison Models (Pretrained - Validation Accuracy):**
    * ResNet-18: 99.90%.
    * EfficientNet-B0: 99.96%.
    * DenseNet121: 99.96%.
    * MobileNetV2: 99.94%.
* **Comparison Models (Randomly Initialized - Validation Accuracy):**
    * ResNet-18: 99.76%.
    * EfficientNet-B0: 99.63%.
    * DenseNet121: 99.80%.
    * MobileNetV2: 99.43%.
* **Comparison Models (Domain Change - Test Accuracy):**
    * ResNet-18: 90.23%.
    * EfficientNet-B0: 83.64%.
    * DenseNet121: 93.25% (Highest).
    * MobileNetV2: 84.51%.

**Key Findings:**
* The custom TraffiKING model achieves reasonable accuracy with a significantly smaller size compared to state-of-the-art models.
* Pretrained models generally outperform the custom model and randomly initialized models, especially in domain change scenarios, highlighting the benefit of transfer learning.
* DenseNet121 showed the best performance on the domain change dataset.

## Usage

Run the `model_train.ipynb` notebook.


## Contact

* [Dominik Barukčić](https://github.com/doms911)
* Nika Božić
* [Borna Josipović](https://github.com/bornajosipovic)
* [Andrija Merlin](https://github.com/nilrema)
* [Vedran Moškov](https://github.com/VMoskov)
* Faculty of Electrical Engineering and Computing, University of Zagreb
* Emails: {dominik.barukcic, nika.bozic, borna.josipovic, andrija.merlin, vedran.moskov} @fer.hr

## References

* GTSRB Dataset: [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
* NumPy: [https://numpy.org/](https://numpy.org/)
* Torchvision: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)
* He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
* Huang, G., et al. (2017). Densely connected convolutional networks. CVPR.
* Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. ICML.
* Sandler, M., et al. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR.
* Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. JMLR.
