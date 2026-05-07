# Titanic Survival Prediction

## Project Overview
This project analyzes Titanic passenger data to identify the key factors affecting survival and builds a Logistic Regression model to predict whether a passenger survived.

## Objectives
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis (EDA)
- Identify important features related to survival
- Build and evaluate a baseline classification model
- Generate a valid Kaggle competition submission

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Notebook

## Dataset
The project uses the **Titanic - Machine Learning from Disaster** dataset from Kaggle. The target variable is `Survived`, where:
- `0` = did not survive
- `1` = survived

## Selected Features
The baseline model uses the following features:
- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`

## Analysis Process
1. Load and inspect the dataset
2. Handle missing values and clean the data
3. Explore survival patterns by sex, class, and age
4. Select relevant features
5. Train a Logistic Regression model
6. Evaluate model performance using accuracy, confusion matrix, and classification report
7. Generate and submit `submission.csv` to Kaggle

## Key Findings
- More passengers did not survive than survived
- Female passengers had a much higher survival rate than male passengers
- First-class passengers had the highest survival rate, while third-class passengers had the lowest
- Age also showed some relationship with survival, with children appearing to have relatively better outcomes

## Model Performance
- **Validation Accuracy:** 0.799
- **Kaggle Submission Score:** 0.76555

## Conclusion
This project shows that sex, passenger class, and age were important factors affecting Titanic survival. The Logistic Regression model achieved a solid baseline performance and produced a valid Kaggle submission. Further improvement could be achieved through more advanced models, feature engineering, or hyperparameter tuning.