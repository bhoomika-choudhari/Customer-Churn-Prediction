Customer Churn Prediction Web App

This is an interactive Streamlit application built to help predict customer churn using machine learning models. It walks you through exploratory data analysis, preprocessing, model training (Random Forest or XGBoost), evaluation, and visualization — all in one place.

What This Project Does
This web app allows you to:

    Visualize churn distribution and key metrics like tenure

    Perform data cleaning and encoding

    Choose between two ML models: Random Forest or XGBoost

    Optionally tune hyperparameters using GridSearchCV

    Evaluate the model using classification metrics and visual tools

    See which features contribute most to churn prediction

    Run cross-validation and visualize the results

Dataset Info
This project uses the Telco Customer Churn dataset.
Make sure the file named WA_Fn-UseC_-Telco-Customer-Churn.csv is placed in the project directory.

You can find the dataset here: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Libraries Used

The project relies on the following Python libraries:

      streamlit
      pandas
      numpy
      matplotlib
      seaborn
      scikit-learn
      xgboost
      joblib

Install everything at once with:

      pip install -r requirements.txt

How to Run This Project
Clone this repository or download the Python file.

Place the dataset CSV in the same folder as your code.

Launch the app locally by running:

      streamlit run app.py

The Streamlit interface will open in your default browser.

Model Options
| Model         | Hyperparameter Tuning | Description                            |
|---------------|------------------------|----------------------------------------|
| Random Forest | Yes (optional)         | Easy to use and good baseline          |
| XGBoost       | Yes (optional)         | Performs better on complex data        |

If a trained model exists (model_Random_Forest.joblib or model_XGBoost.joblib), it will be loaded automatically to save time on reruns.

Visual Outputs
> Churn distribution (bar chart)

> Tenure analysis (boxplot)

> Feature correlation heatmap

> Confusion matrix

> ROC-AUC score

> Cross-validation fold results

> Top 10 most important features

About the Developer

I'm Bhoomika Choudhari, a third-year B.Tech student majoring in Computer Science with a specialization in Artificial Intelligence and Machine Learning (CSE-AIML). This project was built to strengthen my hands-on skills in data preprocessing, machine learning model development, and deploying interactive applications using Streamlit. The main goal was to build a churn prediction tool that is simple, effective, and understandable for both technical and non-technical users.



