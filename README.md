🧠 Tobacco Use Analysis & Prediction Using Machine Learning
A comprehensive data science project that analyzes youth tobacco usage patterns and predicts consumption levels using various supervised machine learning models. The project includes data preprocessing, classification, regression, hyperparameter tuning, and multiple data visualizations to extract insights from a public health dataset.

📊 Overview
This project explores a tobacco usage dataset to:

Predict tobacco usage level using classification models like K-Nearest Neighbors (KNN) and Logistic Regression.

Forecast current tobacco usage percentage with Linear Regression.

Visualize behavioral patterns and feature relationships through detailed plots.

Evaluate model performance using R², accuracy, ROC curves, and confusion matrices.

🧰 Technologies Used
Python 3.x

Pandas for data manipulation

Matplotlib & Seaborn for visualization

Scikit-learn for machine learning models and metrics

📁 Dataset
The dataset (TobaccoDataset.csv) contains data on tobacco usage, awareness, and policy impacts across schools.

Key features include:

Ever tobacco users (%)

Current smokeless tobacco users (%)

Ever e-cigarette use (%)

Awareness about e-cigarette (%)

School heads aware of COTPA, 2003 (%)

Usage_Level (target for classification)

Current tobacco users (%) (target for regression)

⚙️ Model Pipeline
Data Cleaning & Preprocessing

Fill missing values.

Encode categorical target variable.

Scale numerical features.

Machine Learning Models

📌 KNN Classifier: With hyperparameter tuning via GridSearchCV.

📌 Logistic Regression: Multi-class classification with ROC analysis.

📌 Linear Regression: Predicts Current tobacco users (%).

Evaluation Metrics

Accuracy, R² Score, Confusion Matrix, ROC Curve (One-vs-Rest for multi-class).

Visual Explorations

Boxplot, heatmap, pairplot, pie chart, and histogram to understand data distributions and relationships.

📈 Key Results
Best KNN Accuracy: Achieved with tuned parameters.

Logistic Regression Accuracy: Demonstrates strong predictive power.

R² Score (Linear Regression): Measures model fit for tobacco usage percentage.

🖼️ Visualizations Included
📦 Box Plot: E-cigarette use by usage level

🔥 Heatmap: Correlation among features

💠 Pair Plot: Feature interactions by usage level

🍰 Pie Chart: Usage level distribution

📉 Histogram: Distribution of ever tobacco users

🏁 Getting Started
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/tobacco-ml-analysis.git
cd tobacco-ml-analysis
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python PythonProject.py
Make sure the dataset is available in the specified path or update the script accordingly.

📌 To-Do
 Automate preprocessing pipeline

 Add model performance tracking across different datasets

 Export trained models for reuse

🤝 Contributions
Contributions, ideas, and improvements are welcome! Feel free to fork the project and submit a pull request.

