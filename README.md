# HR_Attrition_Model

# 📊 HR Attrition Prediction with Machine Learning
* This project aims to build a machine learning model to predict employee attrition in an organization using HR data. It includes exploratory data analysis (EDA), preprocessing, feature importance, model training, and performance evaluation using both Random Forest and XGBoost classifiers.

# 🗂️ Project Structure

---

# 🧰 Tools & Technologies Used
* Python

* Pandas, NumPy, Seaborn, Matplotlib for data manipulation and visualization

* scikit-learn for ML modeling

* XGBoost for advanced gradient boosting

* Imbalanced-learn for handling class imbalance

* Streamlit for web deployment
---
# 📌 Key Features
* EDA with visual breakdowns of categorical distributions

* Label encoding for categorical features

* Class imbalance handling with RandomOverSampler

**Machine Learning models:**

* Random Forest Classifier

* XGBoost Classifier

**Model evaluation using:**

* Accuracy

* ROC AUC Score

* Confusion Matrix

* Classification Report

* Feature importance visualization

* Streamlit deployment for real-time predictions
---
# 🔍 Dataset Overview
* The dataset WA_Fn-UseC_-HR-Employee-Attrition.csv contains 1,470 employee records with 35 features, including demographics, job role, compensation, and attrition status.

* No missing values

* No duplicates

* Target variable: Attrition (Yes/No)
---

# 📊 EDA Highlights
* Attrition Rate: 16.1% of employees left the company

* Most employees travel rarely and work in R&D

* Male employees are more frequent in the dataset

* High attrition is correlated with:

* OverTime

* JobRole

* JobSatisfaction

*YearsSinceLastPromotion
---

# 🧪 Model Performance

**✅ Random Forest Classifier**
**Metric	Score**
* Train acc   1.00%
* Test acc    0.94%
* Recall   97%
* Precision  	95%
* ROC AUC	0.99

**✅ XGBoost Classifier (With Oversampling)**
**Metric	Score**
* Train acc   1.00%
* Test acc    0.95%
* Recall   98%
* Precision  	94%
* ROC AUC	0.98
---

# 🚀 XGBoost showed better generalization with higher ROC AUC, making it the final model of choice.

**🔥 Top Features Driving Attrition**
**According to feature importance scores:**

* OverTime

* MonthlyIncome

* Age

* TotalWorkingYears

* JobLevel

* YearsSinceLastPromotion

* DistanceFromHome

* EnvironmentSatisfaction
---

# 🧠 Business Recommendations
**Based on the insights and model predictions, the following strategic actions are advised:**

* Monitor and Reduce Overtime:

* Overtime was the most significant predictor of attrition.

* Enforce better work-life balance policies.

**Promotions and Career Growth:**

* Many employees left after long periods without promotion.

* Implement more transparent promotion policies.

**Targeted Retention Strategy:**

* Focus on mid-level employees with low satisfaction and long tenure.

* Use this model to flag at-risk individuals for HR review.

**Salary Review for Key Roles:**

* Attrition is higher among lower salary bands.

* Benchmark salaries with industry standards and adjust if needed.

**Department-Level Interventions:**

* Departments like Sales and R&D show different attrition patterns.

* Create department-specific retention strategies.
---

# 🚀 Running the Streamlit App
**To run the prediction dashboard locally:**

* streamlit run app/streamlit_app.py

--- 
# 💾 Model Export
**The final XGBoost model was saved using:**

* with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
  
**Label encoders were saved as:**

* with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

  ---
  
# 📌 Final Notes
This model is a powerful decision-support tool for HR departments. With proper threshold tuning and business integration, it can help proactively retain talent and reduce churn-related costs.

