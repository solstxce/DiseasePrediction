# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load your dataset
# Ensure to update these paths to where your training and testing datasets are stored
training_data_path = 'Training.csv'
testing_data_path = 'Testing.csv'

# Load training and testing data
train_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(testing_data_path)

# Preprocess your dataset
# Assuming that the last column in your dataset is the target column, and all others are features
X_train = train_data.iloc[:, :-1]  # Features from the training data
y_train = train_data.iloc[:, -1]   # Target column from the training data

X_test = test_data.iloc[:, :-1]    # Features from the testing data
y_test = test_data.iloc[:, -1]     # Target column from the testing data

# Split the training set into train/validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],           # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],          # Depth of each tree
    'min_samples_split': [2, 10, 20],         # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 5, 10],           # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt'],         # Number of features to consider when looking for the best split
}

# Initialize a GridSearchCV object
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # Use all available processors
    verbose=2        # Verbosity mode for details during grid search
)

# Perform grid search to find the best hyperparameters
grid_search.fit(X_train_split, y_train_split)

# Extract the best parameters found by grid search
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the RandomForestClassifier using the best parameters
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train_split, y_train_split)

# Validate the model on the validation set
val_predictions = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy}")

# Test the model on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# Generate and print the classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, test_predictions))

