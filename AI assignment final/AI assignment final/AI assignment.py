import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Read the dataset with the full path
dataset = pd.read_csv('c:\\Users\\Lenovo\\PycharmProjects\\AI assignment final\\AI assignment.py\\breast_cancer.csv')

# Display the first 12 rows
print(dataset.head(12))

# Display the last 5 rows
print(dataset.tail(5))

# Print the number of rows and columns
print(f"Number of rows: {dataset.shape[0]}")
print(f"Number of columns: {dataset.shape[1]}")

# Check for missing values
print(dataset.isnull().sum())

# Extract features (X) and target (Y)
X = dataset.drop('diagnosis', axis=1)  # Features (all columns except 'diagnosis')
Y = dataset['diagnosis']  # Target (the 'diagnosis' column)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, Y_train)

# Predict the target for the test set
Y_pred = log_reg.predict(X_test)

# Display the first 10 actual and predicted values
print("First 10 actual values:", Y_test[:10].values)
print("First 10 predicted values:", Y_pred[:10])

# Compute the Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(Y_test, Y_pred, pos_label='M')
recall = recall_score(Y_test, Y_pred, pos_label='M')
f1 = f1_score(Y_test, Y_pred, pos_label='M')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
