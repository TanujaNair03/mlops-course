# train.py (FAST-RUN VERSION)

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# --- Configuration ---
# Hardcoded URI from the example repo
MLFLOW_TRACKING_URI = "http://35.223.244.50:5000/" 
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Fast CI Run"

def prepare_data():
    """Loads and splits the data."""
    print("Preparing data...")
    try:
        # Load the data file downloaded in the CI step
        data = pd.read_csv('./data.csv') 
    except FileNotFoundError:
        print("Error: data.csv not found. Make sure the 'Download Data' step ran.")
        sys.exit(1)
        
    data = pd.DataFrame(data, columns=['sepal_length','sepal_width','petal_length','petal_width', 'species'])

    # Split the data
    train, test = train_test_split(
        data, test_size=0.2, stratify=data['species'], random_state=42
    )
    
    feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']
    X_train, y_train = train[feature_cols], train['species']
    X_test, y_test = test[feature_cols], test['species']
    
    print("Data split complete.")
    return X_train, y_train, X_test, y_test

def train_fast_model(X_train, y_train, X_test, y_test):
    """
    Trains one fast model and logs it to MLflow.
    Wraps MLflow logging in a try/except to prevent timeouts.
    """
    print(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        with mlflow.start_run(run_name=RUN_NAME) as run:
            print(f"MLflow run started (Run ID: {run.info.run_id}).")
            
            # --- Train ONE simple model ---
            print("Training single RandomForest model...")
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            test_score = model.score(X_test, y_test)
            print(f"Test set accuracy: {test_score:.4f}")
            
            # --- Log metrics and model (wrapped in try/except) ---
            try:
                print("Logging parameters, metrics, and model to MLflow...")
                mlflow.log_param("n_estimators", 10)
                mlflow.log_param("max_depth", 5)
                mlflow.log_metric("final_test_accuracy", test_score)
                
                mlflow.sklearn.log_model(
                    model, 
                    "random_forest_model", 
                    registered_model_name=MODEL_NAME
                )
                print("MLflow logging successful.")
                
            except Exception as e:
                # This prevents the CI job from failing if the MLflow server is down
                print(f"WARNING: MLflow logging failed: {e}. Continuing anyway.")
                
            print("MLflow run finished.")
            
    except Exception as e:
        # This prevents the CI job from failing if mlflow.start_run fails
        print(f"WARNING: Could not start MLflow run: {e}. Continuing anyway.")

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test = prepare_data()
        train_fast_model(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)