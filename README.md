# MNIST Knowledge Distillation with Large and Distil Models
This project demonstrates knowledge distillation, where a larger, complex model (teacher) helps train a smaller, more efficient model (student). The MNIST dataset is used to train both the large model and the distil model, transferring the knowledge from the large model to the smaller one.

# Large Model
The large model is a neural network designed to serve as the teacher in the distillation process. It is larger and more complex, trained on the MNIST dataset to high accuracy.

Architecture:

Input: 784 (flattened 28x28 image pixels)

Hidden Layer 1: 1600 units

Hidden Layer 2: 1600 units

Output: 10 units (corresponding to 10 digit classes)

# Distil Model
The distil model is a smaller version of the large model, designed to be computationally efficient. It learns from both the dataset labels and the predictions of the large model.

Architecture:

Input: 784 (flattened 28x28 image pixels)

Hidden Layer 1: 800 units

Hidden Layer 2: 800 units

Output: 10 units

# Training
Large Model: The large model is trained using standard cross-entropy loss. It aims to achieve high accuracy on the MNIST dataset.

Distil Model: After training the large model, we use the distil model to learn from both the dataset and the softened outputs of the large model. The training combines cross-entropy loss for the true labels and KL-divergence loss for knowledge transfer.
The train.py file contains the training functions:

train_large(): Trains the large model.

train_distil(): Trains the distil model using knowledge distillation.

# Evaluation
The evaluate.py script contains the function to evaluate both the large and distil models on the MNIST test dataset. It calculates the accuracy by comparing the model predictions with the true labels.

# Grid Search
The main.py script includes a grid search functionality for hyperparameter optimization, allowing you to find the best combination of learning rate, momentum, and epochs for the large model. The best parameters are then used to train the distil model.

# Results
Best Hyperparameters: The optimal learning rate, momentum, and epoch count are determined via grid search.

Large Model Performance: The large model achieves high accuracy on the MNIST dataset. Accuracy: 0.9851

Distil Model Performance: After distillation, the smaller distil model approaches the performance of the large model but with fewer parameters and faster inference time. Accuracy: 0.9759
