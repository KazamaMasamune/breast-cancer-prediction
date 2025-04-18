# 🎯 breast-cancer-prediction - Tumor Classification System

**breast-cancer-prediction** is a machine learning system for classifying breast tumors as malignant or benign, developed as a computer science project. Powered by logistic regression on the Wisconsin Breast Cancer Dataset, it achieves 97.37% accuracy. A Flask web app provides a user-friendly interface to input tumor features and receive predictions with confidence scores, supporting medical diagnostics. 🌟

## 📋 Table of Contents
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features
- 📊 **Tumor Classification**: Predicts malignant/benign tumors using logistic regression.
- 🔍 **High Accuracy**: Achieves 97.37% accuracy with strong precision and recall.
- 🌐 **Web Interface**: Flask app allows input of 30 tumor features via a form.
- 📈 **Confidence Scores**: Returns prediction confidence (e.g., “Benign: 92.50%”).
- 💾 **Model Persistence**: Saves trained model and scaler for reliable predictions.
- 🛠️ **Extensible Design**: Supports adding new models or dataset enhancements.

## 🛠️ Technologies
```yaml
Machine Learning:
  - Python: 3.8+ 🐍
  - scikit-learn: 1.0.2 🤖
  - pandas: 1.3.5 📊
  - numpy: 1.21.6 🔢
  - joblib: 1.1.0 💾
Web:
  - Flask: 2.0.1 🌐
Tools:
  - Git: Version control 🔗
  - VS Code: IDE 💻
Environment:
  - pip: Package manager 📦
Planned:
  - TensorFlow: Deep learning models 🧠
  - Docker: Containerized deployment 🚢
