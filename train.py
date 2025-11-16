import os
import sys
import pandas as pd
import numpy as np  # Make sure to import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# --- Configuration ---
# Hardcoded URI from the example repo
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" 
MODEL_NAME = "iris-random-forest"

def poison_data(df, level):
    """
    Randomly corrupts a percentage of the 'species' labels.
    Assumes 'species' is already numeric (0, 1, 2).
    """
    if level == 0.0:
        print("Poison level is 0. No data will be poisoned.")
        return df, 0
    
    n_poison = int(len(df) * level)
    print(f"Poisoning {n_poison} rows ({level*100}%)...")
    
    # Get random indices to poison
    poison_indices = np.random.choice(df.index, n_poison, replace=False)
    
    # Get the number of classes (e.g., 0, 1, 2 for Iris)
    n_classes = df['species'].nunique()
    
    # For each poisoned index, assign a new *random* (but different) label
    for i in poison_indices:
        original_label = df.loc[i, 'species']
        # Generate a new label that is guaranteed to be wrong
        new_label = (original_label + np.random.randint(1, n_classes)) % n_classes
        df.loc[i, 'species'] = new_label
        
    print(f"Data poisoning complete.")
    return df, n_poison

def prepare_data(poison_level):
    """Loads, poisons, and splits the data."""
    print("Preparing data...")
    try:
        data = pd.read_csv('./data.csv') 
    except FileNotFoundError:
        print("Error: data.csv not found. Make sure the 'Download Data' step ran.")
        sys.exit(1)
        
    data = pd.DataFrame(data, columns=['sepal_length','sepal_width','petal_length','petal_width', 'species'])
    
    # --- FIX: CONVERT LABELS TO NUMBERS ---
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    data['species'] = data['species'].map(species_map)
    # ------------------------------------
    
    # --- POISONING STEP ---
    data, n_poisoned = poison_data(data, poison_level)
    # ----------------------

    train, test = train_test_split(
        data, test_size=0.2, stratify=data['species'], random_state=42
    )
    
    feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']
    X_train, y_train = train[feature_cols], train['species']
    X_test, y_test = test[feature_cols], test['species']
    
    print("Data split complete.")
    return X_train, y_train, X_test, y_test, n_poisoned

def train_and_log(X_train, y_train, X_test, y_test, poison_level, n_poisoned):
    """
    Trains one fast model and logs it to MLflow.
    """
    print(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set a dynamic run name
    run_name = f"Poison_Experiment_{int(poison_level*100)}pct"
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            print(f"MLflow run started: {run_name}")
            
            # --- Log Poisoning Parameters ---
            mlflow.log_param("poison_level_percent", poison_level * 100)
            mlflow.log_param("poisoned_rows_count", n_poisoned)
            # ----------------------------------
            
            print("Training single RandomForest model...")
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            test_score = model.score(X_test, y_test)
            print(f"Test set accuracy: {test_score:.4f}")
            
            # --- Log Metrics and Model ---
            mlflow.log_metric("final_test_accuracy", test_score)
            
            mlflow.sklearn.log_model(
                model, 
                "random_forest_model", 
                registered_model_name=MODEL_NAME
            )
            print("MLflow logging successful.")
            
    except Exception as e:
        print(f"WARNING: MLflow logging failed: {e}. Check server connection.")

if __name__ == "__main__":
    try:
        # Read poison level from environment variable, default to 0
        poison_level = float(os.environ.get('POISON_LEVEL', 0.0))
        
        X_train, y_train, X_test, y_test, n_poisoned = prepare_data(poison_level)
        train_and_log(X_train, y_train, X_test, y_test, poison_level, n_poisoned)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
