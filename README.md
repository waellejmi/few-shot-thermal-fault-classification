# Few-Shot Learning for Thermal Image Classification of Induction Motors

This repository contains the implementation of a Few-Shot Learning (FSL) model for classifying thermal images of induction motors in normal and faulty conditions. The project uses Prototypical Networks with a ResNet18 backbone to achieve effective classification with limited training examples.

## Overview

This project was my first approach to Few-Shot Learning as a student, implementing a solution that can classify thermal images of induction motors with minimal examples per class. The code is structured in a modular way for clarity and educational purposes.

> **Note:** The original academic report for this project is in French. I've included it in the repository for those interested in the detailed methodology and findings.

## Dataset

This project uses the "Thermal Images of Induction Motor" dataset from Kaggle:
- [Dataset Link](https://www.kaggle.com/datasets/amirberenji/thermal-images-of-induction-motor)

The dataset contains thermal images of induction motors in various conditions, including:
- Normal operation
- Multiple types of stator short circuits (at 10%, 30%, and 50% severity)
- Cooling failures
- Rotor defects

To use this project, download the dataset and place it in the `data/` directory.

## Approach

The implementation uses Prototypical Networks, a metric-based meta-learning approach for few-shot classification. Key features include:

- **Embedding model**: Modified ResNet18 with partial layer freezing
- **Episode-based training**: N-way, K-shot, Q-query methodology 
- **Data augmentation**: Extensive augmentation pipeline to improve generalization
- **Learning rate scheduling**: Cosine annealing scheduler for optimization

## Testing the Model

You can test the trained model in two ways:

### 1. Google Colab

A notebook is available for easy testing without local setup requirements:
- [Open in Colab](https://colab.research.google.com/drive/1MRQpNeDNyf9UlkfFFWMt2FKMREf9MGxe?usp=sharing)

To use it:
1. Copy these two files to your Google Drive:
   - `Resnet18_RetrainedV2.pth`: The final model weights
   - `support_embeddings.pt`: Reference embeddings generated from support images
   
2. In the notebook, load them with:
   ```python
   !cp "/content/drive/MyDrive/FSL/Resnet18_RetrainedV2.pth" .
   !cp "/content/drive/MyDrive/FSL/support_embeddings.pt" .
   ```

3. The notebook handles the rest: loading data, inference, and visualizing results.

### 2. Local Testing

Run the prediction script:
```bash
python pred.py
```
Make sure the dataset is in the `data/` directory.

## Key Findings

- Prototypical Networks proved effective for this thermal image classification task
- Data augmentation was crucial for model performance
- Partial unfreezing of the ResNet18 backbone improved results
- The model achieved approximately 83% accuracy with limited training data

## Conclusion

This project demonstrates how Few-Shot Learning can be applied to industrial fault detection scenarios where labeled data might be limited. While implemented as a student project (not by an expert), the approach and code may serve as a helpful starting point for others interested in Few-Shot Learning applications.

For more detailed information about the methodology, experiments, and results, please refer to the full report in the repository (in French).