import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load training data from Training.csv
df_train = pd.read_csv('Training.csv')

# Load testing data from Testing.csv
df_test = pd.read_csv('Testing.csv')

# Feature columns (All columns except the last one, which is the 'prognosis')
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]  # 'prognosis' column

# Preprocess the testing data similarly
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# Split the training data for validation (optional)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier (or any other classifier)
model = RandomForestClassifier()

# Train the model on training data
model.fit(X_train_split, y_train_split)

# Predict on the validation set
val_predictions = model.predict(X_val)

# Print validation accuracy
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions)}")

# Evaluate the model on test data
test_predictions = model.predict(X_test)

# Print test accuracy and classification report
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions)}")
print(classification_report(y_test, test_predictions))

# Export the trained model using joblib
joblib.dump(model, 'disease_prediction_model.joblib')

print("Model exported to 'disease_prediction_model.joblib'")

