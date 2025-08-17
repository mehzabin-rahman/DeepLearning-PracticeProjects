# DeepLearning-PracticeProjects

![Python](https://img.shields.io/badge/python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![Status](https://img.shields.io/badge/status-active-success)

Welcome to my **Deep Learning Practice Projects** repository!  
This repo contains my experiments, exercises, and small projects as I learn and explore different deep learning techniques using Python, TensorFlow, and Keras.

## 📂 Repository Structure
DL-Practice-Projects/
├── SLP # Single-Layer Perceptron on Iris dataset
├── utils# Helper functions, reusable code
└── README.md # This main overview file

> Each folder may contain code, instructions, and optionally a mini README specific to that project. 


## 🚀 Getting Started

1. **Clone the repository**
   git clone https://github.com/<your-username>/DeepLearning-PracticeProjects.git
   cd DeepLearning-PracticeProjects

3. Install dependencies
   pip install -r requirements.txt

4. Run a project

Navigate into a project folder (e.g., SLP) and follow its instructions to train models and make predictions.

⚡ Example: SLP
Trains a Single-Layer Perceptron on the first 2 features of the Iris dataset.

Classifies Setosa vs. Versicolor.

Predict new samples after training:

import numpy as np

new_data = np.array([[5.0, 3.5], [4.6, 3.0]])
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
predicted_classes = (predictions > 0.5).astype(int)
print(predicted_classes)


📌 Notes
Each project is self-contained.

Some projects are small experiments; others include training scripts and example outputs.

Focused on learning and practice, not production-ready code.

🛠 Skills & Topics Covered (soon)
Neural networks: SLP, MLP, CNN, RNN, GNN, NLP, GRU, Transformers, Stable Diffusion

Data preprocessing: scaling, normalization

Optimizers: SGD, Adam, AdamW

Loss functions: Binary/Categorical Crossentropy, MSE, MAE

Model evaluation, prediction, and experimentation

✨ Author
Mehzabin Rahman Portia

📫 Feedback / Contributions
Personal learning repository, but suggestions or tips are welcome!
