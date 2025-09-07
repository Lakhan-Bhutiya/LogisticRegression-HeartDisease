# Logistic Regression (from scratch) 


**Repository / Notebook:** `LogisticRegreesion.ipynb`


## 📌 Overview


This project implements **logistic regression from scratch** without using scikit-learn or any other high-level ML library. All components of the model — hypothesis (sigmoid), cost function, gradient descent optimization, regularization, and evaluation metrics — are manually implemented using only NumPy and pandas.


The goal is to demonstrate the mechanics behind logistic regression and provide a fully transparent, educational implementation.


---

## 🔹 Intuition

- Linear regression predicts a continuous value.  
- Logistic regression takes a linear equation and passes it through the **sigmoid function** to squash the output into a probability range [0, 1].  
- This makes it suitable for classification.

---

## 📘 Logistic Regression Theory

### 1. Model Equation


<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\hat{y}=\sigma(WX+b)\quad\text{where}\;\sigma(z)=\frac{1}{1+e^{-z}}" />
</p>



- **X** → Input features (like blood pressure, BMI, smoking, etc.).  
- **W** → Weights (importance of each feature).  
- **b** → Bias term.  
- **\\(\\hat{y}\\)** → Predicted probability (between 0 and 1).  
- **Sigmoid (\\(\\sigma\\))** → Converts any number into probability.

---


---

### 2. Gradient Descent Updates

$$
W := W - \\alpha \\cdot \\frac{\\partial L}{\\partial W}, 
\\quad 
b := b - \\alpha \\cdot \\frac{\\partial L}{\\partial b}
$$

- **\\(\\alpha\\)** → Learning rate (step size).  
- Gradient tells us how much to adjust weights to reduce error.

---

### 3. Prediction Rule

$$
\\hat{y}_{class} =
\\begin{cases}
1 & \\text{if } \\hat{y} \\geq 0.5 \\\\
0 & \\text{otherwise}
\\end{cases}
$$

---

---

## ⚙️ Why Scaling and Train/Test Split?

### 🔹 Scaling
- Ensures features are on the same scale (e.g., income vs. age).  
- Helps **gradient descent converge faster**.  
- Prevents numerical issues in sigmoid (overflow/underflow).  

### 🔹 Train/Test Split
- Training set → model learns weights.  
- Test set → checks if model generalizes.  
- Prevents **overfitting** (memorizing training data).  

---
