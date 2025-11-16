import os
import subprocess
import time

# Define the poison levels to test
POISON_LEVELS = [0.05, 0.10, 0.50] 

print("--- Starting Data Poisoning Experiments ---")

for level in POISON_LEVELS:
    print(f"\n--- Running experiment for {level*100}% Poisoning ---")

    # Set the poison level as an environment variable
    os.environ['POISON_LEVEL'] = str(level)

    # Run the train.py script
    # We use subprocess.run to ensure it runs as a fresh process
    subprocess.run(["python", "train.py"])

    print(f"--- Finished {level*100}% experiment. ---")
    time.sleep(2) # Pause for 2 seconds

print("\nAll poisoning experiments complete. Check your MLflow UI.")
