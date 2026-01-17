#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# EDA
df.isnull().sum()


# In[8]:


# Check for missing values (note: 0s may represent missing in medical data like Glucose, BMI)
print("\nZero values (potential missing):")
print(df.eq(0).sum())


# In[9]:


# Visualize distributions
plt.figure(figsize=(10,8))
for i, col in enumerate(['Glucose', 'BloodPressure', 'BMI', 'Insulin'], 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[10]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, center=0)
plt.title('Feature Correlation Matrix')
plt.show()


# In[11]:


# Replace zero values (medical implausible = missing) with median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)


# In[12]:


# Box plots for key features by Outcome
key_features = ['Glucose', 'BMI', 'BloodPressure', 'Age']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(key_features):
    row, col = i // 2, i % 2
    sns.boxplot(x='Outcome', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Outcome')
plt.tight_layout()
plt.show()


# In[13]:


# Outlier capping
def outlier_capping(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_extreme = Q1 - 1.5 * IQR
    upper_extreme = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_extreme, upper=upper_extreme)
    return df

num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in num_cols:
    df = outlier_capping(df, col)

print("After preprocessing - describe:")
print(df.describe())


# In[14]:


# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[15]:


# Scale features
std_sca = StandardScaler()
X_train_scaled = std_sca.fit_transform(X_train)
X_test_scaled = std_sca.transform(X_test)


# In[16]:


# Train Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[17]:


# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
print("Predictions:", y_pred)
print("Prediction Probabilities:", y_pred_proba)


# In[18]:


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")


# In[19]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[20]:


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[21]:


# Coefficients
feature_names = X.columns
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)
print(coef_df)


# In[22]:


# Visualize coefficients
plt.figure(figsize=(10, 8))
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title('Logistic Regression Coefficients')
plt.axvline(0, color='black', linestyle='--')
plt.show()


# In[23]:


file='log7.pkl'


# In[24]:


pickle.dump(model,open(file,'wb'))


# In[25]:


scale= 'scale.pkl'


# In[26]:


pickle.dump(std_sca,open(scale,'wb'))


# Interview Questions:
# 1. What is the difference between precision and recall?

# Precision measures how many predicted positives are actually correct
# Precision = TP/(TP+FP)
# 
# recall measures how many actual positives the model identifies
# Recall = TP/(TP+FN)
# 
# The difference between precision and recall:
# 
# Precision focuses on prediction quality: of all diabetes cases predicted, what % are actually diabetic? High precision minimizes false alarms
# 
# Recall focuses on detection: of all true diabetics, what % did model catch? High recall catches most cases
# 
# Diabetes example: 
# Model predicts 100 positive. Precision 80% = 80 real diabetics. But if 200 actual diabetics exist, recall = 80/200 = 40%
# 
# Use precision when false positives hurt more (spam filter). Use recall when missing positives catastrophic (cancer detection).

# 2. What is cross-validation, and why is it important in binary classification?

# Cross-validation(CV) splits data into K folds, trains on K-1, tests on 1, rotates. Averages performance across folds.
# 
# CVimportance in binary classification:
# 
# -Single train-test split risks luck (diabetes data: 65% non-diabetic, 35% diabetic)
# 
# -CV gives robust estimate of true performance
# 
# -Catches overfitting (model memorizes train, fails test)
# 
# -Handles class imbalance better
# 
# Cross-Validation(CV) validates model generalizes beyond one lucky split.
