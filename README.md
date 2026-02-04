# ML Classification Model Evaluation

## Problem Statement

The objective of this project is to design, implement, and evaluate multiple machine learning classification models on a single dataset. The goal is to compare model performance using standard evaluation metrics and identify the most effective model for the given classification problem.

---

## Dataset Description

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.  
The dataset consists of **569 samples** and **30 numerical features** derived from digitized images of fine needle aspirates (FNA) of breast masses.

- **Target variable**
  - `M` – Malignant
  - `B` – Benign
- The dataset contains no missing values.
- This is a binary classification problem.

---

## Models Used

The following machine learning models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

## Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy  
- AUC (Area Under the ROC Curve)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|---------|-----|----------|--------|----------|-----|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## Observations

| ML Model Name | Observation |
|--------------|------------|
| Logistic Regression | Performed well due to effective feature scaling and linear separability of the dataset. |
| Decision Tree | Simple and interpretable but showed slightly lower generalization performance. |
| kNN | Achieved good results after scaling; performance depends on distance metrics and parameter selection. |
| Naive Bayes | Fast and efficient but exhibited lower recall due to the independence assumption. |
| Random Forest (Ensemble) | Improved accuracy and robustness by combining multiple decision trees. |
| XGBoost (Ensemble) | Delivered the best overall performance by capturing complex patterns using gradient boosting. |

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

- Upload test datasets in CSV format  
- Select one or more trained models  
- Evaluate model performance on new data  
- View evaluation metrics, confusion matrix, and classification report  
- Rebuild and retrain models when required  

---

## Conclusion

The comparative analysis shows that ensemble models outperform individual classifiers on this dataset. Among all evaluated models, **XGBoost** demonstrated the most consistent and robust performance across all evaluation metrics.
