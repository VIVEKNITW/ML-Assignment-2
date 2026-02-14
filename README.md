# Machine Learning Classification Web Application

## 1. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a real-world dataset. 

The project includes:
- Implementation of 6 classification algorithms
- Evaluation using multiple performance metrics
- Comparative analysis of model performance
- Deployment of the best-performing models using Streamlit

This project demonstrates the complete end-to-end machine learning workflow including model development, evaluation, UI creation, and cloud deployment.

---

## 2. Dataset Description

Dataset Used: **Breast Cancer Wisconsin Dataset**

Source: Scikit-learn built-in dataset  
Type: Binary Classification  

- Total Instances: 569
- Total Features: 30 numerical features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

The dataset consists of features computed from digitized images of breast mass cell nuclei. The task is to classify whether a tumor is malignant or benign.

---
## 3. Data Preprocessing

- Performed train-test split (80% training, 20% testing)
- Applied StandardScaler for feature normalization
- Saved trained scaler for consistent transformation during deployment
- Ensured same preprocessing pipeline used in Streamlit application

--

## 4. Machine Learning Models Implemented

The following six models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

All models were trained using an 80-20 train-test split and evaluated using identical metrics for fair comparison.

---

## 5. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 6. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.973684 | 0.997380 | 0.972222 | 0.985915 | 0.979021 | 0.943898 |
| Decision Tree | 0.938596 | 0.932362 | 0.944444 | 0.957746 | 0.951049 | 0.868860 |
| KNN | 0.947368 | 0.981985 | 0.957746 | 0.957746 | 0.957746 | 0.887979 |
| Naive Bayes | 0.964912 | 0.997380 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| Random Forest | 0.964912 | 0.996069 | 0.958904 | 0.985915 | 0.972222 | 0.925285 |
| XGBoost | 0.956140 | 0.990829 | 0.958333 | 0.971831 | 0.965035 | 0.906379 |


---

## 7. Observations on Model Performance

| ML Model | Observation |
|------------|-------------|
| Logistic Regression | Performed strongly due to linear separability in feature space. |
| Decision Tree | Achieved good accuracy but showed slight tendency to overfit. |
| KNN | Sensitive to feature scaling but performed well after normalization. |
| Naive Bayes | Provided competitive results despite strong independence assumptions. |
| Random Forest | Improved generalization performance due to ensemble averaging. |
| XGBoost | Delivered high performance due to gradient boosting optimization and regularization. |

Overall, ensemble methods (Random Forest and XGBoost) showed better stability and predictive power compared to individual classifiers.

---

## 8. Project Structure

```
ml-assignemt-2/
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│   ├── Logistic Regression.pkl
│   ├── Decision Tree.pkl
│   ├── KNN.pkl
│   ├── Naive Bayes.pkl
│   ├── Random Forest.pkl
│   ├── XGBoost.pkl
│   └── scaler.pkl
│
├── test_data/
│   └── sample_test_data.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 9. How to Run the Application Locally
### Step 1: Install Dependencies
pip install -r requirements.txt

### Step 2: Run Streamlit Application
```
streamlit run app.py
```
The application will open in your browser at:

http://localhost:8501

---

## 10. Streamlit App Features

- Upload test dataset (CSV format)
- Model selection dropdown
- Display evaluation metrics
- Display confusion matrix
- Display prediction output

---

## 11. Deployment Details

GitHub Repository Link:
[text](https://github.com/VIVEKNITW/ML-Assignment-2)

Live Streamlit App Link:
(Insert your deployed Streamlit Community Cloud link here)

---

## 12. Conclusion

This project successfully compares multiple classification algorithms on a real-world dataset and demonstrates the complete machine learning lifecycle including model development, evaluation, and deployment.

Ensemble models generally showed superior performance, highlighting the power of combining multiple weak learners into a strong predictive model.


## 13. Author

Name: Vivekanand Shanbhag  
Course: M.Tech (AI/ML)
Bits ID: 2025aa05717
Assignment: Machine Learning – Assignment 2  