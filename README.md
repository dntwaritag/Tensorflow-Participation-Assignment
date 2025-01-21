# Neural Network with TensorFlow: Dataset Loading, and Training

## **Project Overview**
This project implements a neural network using TensorFlow to load, preprocess, train, and evaluate a dataset. The work includes visualization of results and predictions. The implementation satisfies all the provided requirements, including hidden layers, accurate class handling, and appropriate data processing.

## **Team Members**
- **Denys Ntwaritaganzwa**
- **Ruth Iradukunda**

---

## **Objectives**
- Load a dataset and preprocess the data for training.
- Build a neural network with at least one hidden layer containing 128 neurons.
- Ensure the output layer matches the number of classes in the dataset.
- Train the model using appropriate loss functions, optimizers, and metrics.
- Save the trained model and make predictions.
- Visualize the results and provide proof of participation.

---

## **Features**
1. **Data Loading**: The MNIST dataset is loaded using TensorFlow's built-in datasets.
2. **Preprocessing**: 
   - Features (`X`) are normalized.
   - Labels (`Y`) are one-hot encoded.
3. **Model Structure**:
   - Input layer.
   - One hidden layer with 128 neurons and ReLU activation.
   - Output layer with `softmax` activation for classification.
4. **Training**:
   - The model is compiled with `SparseCategoricalCrossentropy` loss.
   - Optimizer: `Adam`.
   - Metrics: `Accuracy`.
5. **Evaluation**: Model performance is evaluated on test data.
6. **Visualization**:
   - Visualization of predictions for image data (MNIST samples).
   - Confusion matrix for classification results.
7. **Model Saving**:
   - Trained model is saved for reuse.

---

## **Dataset**
The MNIST dataset is used in this project. It consists of:
- **Training data**: 60,000 grayscale images of handwritten digits (28x28 pixels).
- **Test data**: 10,000 grayscale images for evaluation.
- **Classes**: 10 digits (0-9).

---

## **Instructions**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-link.git
   ```
2. **Navigate to the Project Folder**:
   ```bash
   cd your-project-folder
   ```
3. **Install Dependencies**:
   Ensure TensorFlow, NumPy, Pandas, Matplotlib, and Seaborn are installed:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn
   ```
4. **Run the Script**:
   Execute the main Python script:
   ```bash
   python neural_network.py
   ```
5. **View Results**:
   - Check predictions in the output.
   - Examine model summary, accuracy, and loss metrics in the console.
   - Review saved model files for reuse.

## **Proof of Participation**
GitHub repository includes contributions from all team members.
Commit history and version control demonstrate collaboration.
![image](https://github.com/user-attachments/assets/10d69002-8555-49a9-ba72-104734d6c113)

---
## **Acknowledgments**
This project was completed as part of the neural network assignment in TensorFlow. Special thanks to the course instructor and teammates for their collaboration.
