import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, r2_score,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier

# Load and preprocess dataset
df = pd.read_csv("D:/c drive downloads/TobaccoDataset.csv", na_values=["", "NA", "N/A"])
df.columns = df.columns.str.strip()

# Fill missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col == "Students who noticed anti-tobacco messages (%)":
            df[col] = df[col].fillna(0)
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

# Encode Usage_Level (for classification)
label_encoder = LabelEncoder()
df["Usage_Level_Encoded"] = label_encoder.fit_transform(df["Usage_Level"])

# ✅ Features (excluding regression target to avoid data leakage)
features = [
    'Ever tobacco users (%)',
    'Ever smokeless tobacco users (%)',
    'Current smokeless tobacco users (%)',
    'Ever e-cigarette use (%)',
    'Awareness about e-cigarette (%)',
    'School heads aware of COTPA, 2003  (%)'
]

# Prepare data
X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_class = df["Usage_Level_Encoded"]
y_reg = df["Current tobacco users (%)"]

# Train-test split
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ KNN with Hyperparameter Tuning (Usage_Level) ------------------
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_class_train)
best_knn = grid_search.best_estimator_
y_class_pred_knn = best_knn.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_class_test, y_class_pred_knn)
print(f"Best KNN Parameters: {grid_search.best_params_}")
print(f"KNN Classification Accuracy (Usage Level): {knn_accuracy:.2f}")

# Confusion matrix for KNN
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_class_test, y_class_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues')
plt.title("KNN Classification: Usage Level")
plt.tight_layout()
plt.show()

# ------------------ Linear Regression (Fixed: no target leakage) ------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_reg_train)
y_reg_pred = lr.predict(X_test_scaled)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f"Linear Regression R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_reg_test, y=y_reg_pred, color='teal', edgecolor='black')
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel("Actual Current Tobacco Users (%)")
plt.ylabel("Predicted")
plt.title(f"Linear Regression: Predicting Tobacco Use\nR² = {r2:.2f}")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ Logistic Regression (Usage_Level) ------------------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_class_train)
y_class_pred_log_reg = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_class_test, y_class_pred_log_reg)
print(f"Logistic Regression Accuracy (Usage Level): {log_reg_accuracy:.2f}")

plt.figure(figsize=(6, 5))
cm_log_reg = confusion_matrix(y_class_test, y_class_pred_log_reg)
disp_log_reg = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg, display_labels=label_encoder.classes_)
disp_log_reg.plot(cmap='Blues')
plt.title("Logistic Regression: Usage Level")
plt.tight_layout()
plt.show()

# ------------------ ROC Curve (Multiclass, Usage_Level) ------------------
y_class_binarized = label_binarize(y_class_test, classes=[0, 1, 2])
log_reg_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
log_reg_ovr.fit(X_train_scaled, y_class_train)
y_score = log_reg_ovr.predict_proba(X_test_scaled)

plt.figure(figsize=(8, 6))
colors = ['darkorange', 'blue', 'green']
for i in range(3):
    fpr, tpr, _ = roc_curve(y_class_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label_encoder.classes_[i]} ROC curve (area = {roc_auc:.2f})', color=colors[i])

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression (Usage Level)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ------------------ VISUALIZATION SECTION ------------------


# 1. Box plot: Ever e-cigarette use (%) by Usage Level
plt.figure(figsize=(8, 5))
sns.boxplot(x="Usage_Level", y="Ever e-cigarette use (%)", data=df, palette="Set2")
plt.title("Ever E-Cigarette Use by Usage Level")
plt.xlabel("Usage Level")
plt.ylabel("Ever E-Cigarette Use (%)")
plt.tight_layout()
plt.show()

# 2. Heatmap: Correlation matrix of numerical features
plt.figure(figsize=(10, 8))
corr = df[features + ['Current tobacco users (%)']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# 3. Pair plot: Selected features by Usage Level
sns.pairplot(df, vars=[
    'Ever tobacco users (%)',
    'Ever smokeless tobacco users (%)',
    'Ever e-cigarette use (%)'
], hue='Usage_Level', palette='husl')
plt.suptitle("Pair Plot of Key Features by Usage Level", y=1.02)
plt.show()

# 4------------------ PIE CHART: Usage Level Distribution ------------------
plt.figure(figsize=(6, 6))
usage_counts = df['Usage_Level'].value_counts()
colors = ['#66b3ff', '#99ff99', '#ff9999']  # Visually appealing colors
plt.pie(usage_counts, labels=usage_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Proportion of Usage Levels in Dataset", fontsize=14)
plt.axis('equal')  # Ensures it's a perfect circle
plt.tight_layout()
plt.show()

# 5------------------ HISTOGRAM: Ever Tobacco Users (%) ------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['Ever tobacco users (%)'], bins=15, kde=True, color='coral', edgecolor='black')
plt.title("Distribution of Ever Tobacco Users (%)")
plt.xlabel("Ever Tobacco Users (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


