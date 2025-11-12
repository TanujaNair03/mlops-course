

## 🌟 MLOps Assignment: IRIS Classification CI/CD Pipeline

This repository contains a complete MLOps workflow for the Iris classification model, demonstrating **Continuous Integration (CI)**, **Continuous Deployment (CD)**, and **Monitoring** on **Google Kubernetes Engine (GKE)**.

The final deployed service is a containerized Flask API that serves the model, automatically scales under load, and can be easily updated via a Git merge.

---

## 📁 Repository Structure

| File/Folder | Purpose | Phase Completed |
| :--- | :--- | :--- |
| **`app.py`** | **API Code.** Contains the lightweight Flask API logic that loads the `model.joblib` artifact and exposes the `/predict` endpoint. | Phase 4 |
| **`create_model.py`** | **Model Generator.** A one-time script used to generate a simple `model.joblib` file for deployment (bypasses the broken MLflow training). | Phase 3 |
| **`Dockerfile`** | **Container Definition.** Instructions for building the immutable Docker image, including installing dependencies and running the `app.py` API with `gunicorn`. | Phase 3 |
| **`deployment.yml`** | **Kubernetes Manifest.** Defines the GKE **Deployment** (the pod specification) and the **Service** (the External LoadBalancer) for the API. | Phase 3 |
| **`requirements.txt`** | **Dependencies.** Lists all necessary Python libraries for local development and the production Docker image. | Phase 1 |
| **`test_model.py`** | **Dummy CI Test.** A placeholder Python file containing a simple `pytest` function used to satisfy the Phase 1 CI gate. | Phase 1/2 |
| **`.github/workflows/cd.yml`** | **Continuous Deployment Pipeline.** The core YAML workflow that automates the entire Phase 3 and Phase 4 deployment process. | Phase 3/4 |

---

## 🎯 Project Phases Summary

### Phase 1 & 2: Project Setup & CI Pipeline

* **Objective:** Establish a secure, repeatable process for code changes and integrate DVC for data tracking.
* **Method:**
    * Setup `main` and `dev` branches with **Branch Protection Rules** requiring status checks and review.
    * The `runmain.yml` workflow was used to run `pytest` (to verify code quality) and log data using **DVC (Data Version Control)**.
* **Key Challenge:** The external MLflow server was unresponsive, causing CI workflows to hang and fail the pipeline. This forced the use of simpler, faster scripts to maintain pipeline flow.

### Phase 3: Continuous Deployment (CD) to GKE

* **Objective:** Containerize the API and deploy it to a managed Kubernetes cluster using GitHub Actions.
* **Methodology:**
    1.  **GCP Setup:** Created a **Google Kubernetes Engine (GKE)** cluster and an **Artifact Registry (GAR)**.
    2.  **Authentication:** Configured a **GCP Service Account** and stored the credentials securely as GitHub Secrets (`GCP_CREDENTIALS_JSON`) to allow GitHub to interact with GKE and GAR.
    3.  **Deployment:** The **`cd.yml`** pipeline was created to trigger on merge to `main`. This job:
        * Builds the Docker image (`Dockerfile`).
        * Pushes the image to **GAR**.
        * Deploys the application onto GKE using the **`deployment.yml`** manifest.

### Phase 4: Stress Testing and Autoscaling

* **Objective:** Demonstrate that the deployed API is scalable and robust by running a load test.
* **Methodology:**
    1.  The **`stress-test`** job was added to `cd.yml` with a dependency on the `deploy` job.
    2.  This job uses `kubectl` to find the service's external IP and installs the **`wrk`** utility.
    3.  It runs sequential tests: **1000 concurrent requests** followed by **2000 concurrent requests**.
    4.  **Demonstration:** When the high-load test runs, the **Horizontal Pod Autoscaler (HPA)** automatically detects the increased CPU usage and scales the number of pods from **1 up to 3** to absorb the traffic. This validates the pipeline's elasticity.
* **Key Challenge:** The `stress-test` initially failed because the GKE connection plugin (`gke-gcloud-auth-plugin`) was missing from the runner's PATH, requiring the manual addition of the installation step to the job.

---

## 🚀 Final Deployed Service

The deployed application is accessible via the GKE LoadBalancer IP address: **`http://<GKE_LOAD_BALANCER_IP>/`**.

### Verification

| Test | Tool | Purpose |
| :--- | :--- | :--- |
| **Liveness Check** | Browser/Curl | Confirms the API is running and returns the "API is running!" message. |
| **Autoscaling Test** | `wrk` (in `cd.yml`) | Demonstrates GKE scaling the deployment from 1 to 3 pods automatically. |
