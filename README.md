# 💳 Credit Card Fraud Detection using Machine Learning

This project uses machine learning techniques to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 📌 Project Objective
- Detect fraudulent transactions with high precision and recall.
- Handle data imbalance using techniques like SMOTE.
- Evaluate model performance using classification metrics.

## 📊 Tools & Libraries
- Python, NumPy, Pandas
- Scikit-learn, Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)

## 📁 Dataset
> Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
> It contains 284,807 transactions with only 492 fraud cases (~0.17%).

## 📈 Algorithms Used
- Logistic Regression
- Random Forest Classifier
- SMOTE for oversampling

## ⚙️ How to Run
1. Clone the repo
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Download the dataset and place it inside `dataset/` folder.
4. Open the notebook:  
   `notebooks/fraud_detection.ipynb`

## 📌 Results
- F1-score: ~0.90+
- Precision: High
- Model was trained using SMOTE-balanced data for better fraud detection.

## 📷 Visuals
![Confusion Matrix](images/roc_curve.png)

---

## 🙌 Author
Built with ❤️ by [Your Name]  
📧 [your.email@example.com] | [LinkedIn](#) | [GitHub](#)
