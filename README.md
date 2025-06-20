# Customer Churn Prediction: End-to-End MLOps Project

This project demonstrates a complete machine learning operations (MLOps) pipeline for predicting customer churn, from data handling to model deployment and monitoring. It's split into two main parts:

1.  **Machine Learning Pipeline (`customer-churn-prediction` directory):** This part focuses on training and evaluating a model to predict customer churn.
2.  **Web Application (`churn-prediction-webapp` directory):** This is a simple web interface that allows users to input customer data and get a churn prediction using the deployed model.

## Project Goal

The primary goal is to predict whether a customer is likely to "churn" (stop using a service or cancel a subscription) based on their demographic information, account details, and service usage. This allows businesses (like telecom, SaaS, streaming services) to proactively identify at-risk customers and take retention actions.

## 1. Machine Learning Pipeline (`customer-churn-prediction/`)

This part of the project handles everything related to data, model training, and preparing the model for use.

### Key Steps:

1.  **Data Ingestion:**
    *   Loads the customer data from a CSV file (e.g., the Telco Customer Churn dataset from Kaggle).
    *   Performs a basic initial look at the data, like checking for missing values and understanding the different types of information (numbers, text).

2.  **Data Preprocessing:**
    *   **Cleaning:** Handles any missing or incorrect data (e.g., `TotalCharges` that might be empty).
    *   **Encoding:** Converts categorical features (like "Yes/No", "Male/Female", "Contract Type") into numerical representations that machine learning models can understand. This project uses:
        *   **Label Encoding:** For features with a natural order or only two categories.
        *   **One-Hot Encoding:** For features with multiple categories without a natural order, creating new binary (0/1) columns for each category.
    *   **Scaling:** Normalizes numerical features (like `tenure`, `MonthlyCharges`) so they are on a similar scale. This helps some algorithms perform better.
    *   **Splitting:** Divides the dataset into three parts:
        *   **Training set:** Used to teach the model.
        *   **Validation set (optional):** Used to tune model parameters during development.
        *   **Test set:** Used to evaluate the final model's performance on unseen data.
    *   **Saving Preprocessor:** The steps taken to preprocess the data (like how to scale numbers or encode categories) are saved into a file (`preprocessor.pkl`). This is crucial so that new data for prediction can be transformed in exactly the same way.

3.  **Feature Engineering:**
    *   Creates new, potentially more informative features from existing ones. Examples:
        *   **Tenure Buckets:** Grouping `tenure` (how long someone has been a customer) into categories like "0-1 year", "1-2 years", etc.
        *   **Service Count:** Counting how many additional services (like Online Security, Tech Support) a customer uses.
        *   **Interaction Terms:** Combining two features to capture potential combined effects (e.g., "Contract Type" x "Payment Method").

4.  **Model Training:**
    *   Uses the preprocessed training data to "teach" machine learning models.
    *   **Models Used:**
        *   **XGBoost / LightGBM:** Powerful gradient boosting algorithms, often providing high accuracy (used as the primary model here).
        *   **Logistic Regression:** A simpler, more interpretable model, good for understanding feature importance.
    *   **Cross-Validation:** A technique used during training to get a more robust estimate of how well the model will perform on unseen data.
    *   **Metric:** Area Under the ROC Curve (AUC) is a common metric for classification tasks, especially with imbalanced datasets (like churn, where fewer customers might churn than not).
    *   **MLflow Tracking:** All training runs, parameters, metrics, and model artifacts are logged using MLflow. This helps in tracking experiments and reproducing results.
        *   You can view these logs by running `mlflow ui` in the terminal from the `customer-churn-prediction` directory.

5.  **Model Evaluation:**
    *   Assesses the trained model's performance on the unseen test data.
    *   **Metrics Used:**
        *   **Accuracy:** Overall correctness.
        *   **Precision:** Of those predicted to churn, how many actually churned.
        *   **Recall:** Of all those who actually churned, how many did the model correctly identify.
        *   **F1-Score:** A balance between precision and recall.
        *   **ROC-AUC:** Measures the model's ability to distinguish between churners and non-churners.
    *   **Confusion Matrix:** A table showing correct and incorrect predictions for each class.
    *   **Feature Importance:** (For models like XGBoost) Shows which features had the most impact on the model's predictions.

6.  **Model Export & Serving:**
    *   The best trained model (e.g., `churn_model_xgb.pkl`) and the preprocessor (`preprocessor.pkl`) are saved using `joblib`.
    *   An API (Application Programming Interface) is created using **FastAPI**. This API:
        *   Loads the saved model and preprocessor.
        *   Provides an endpoint (e.g., `/predict`) that can receive new customer data.
        *   Preprocesses the new data using the saved preprocessor.
        *   Uses the loaded model to make a churn prediction.
        *   Returns the prediction (e.g., "Yes" or "No" for churn, and the probability).

7.  **CI/CD Pipeline (GitLab CI/CD):**
    *   Automates the process of testing, building, and (conceptually) deploying the application.
    *   **Steps typically include:**
        1.  **Setup:** Prepares the environment, installs dependencies.
        2.  **Build Artifacts:**
            *   Runs `preprocess.py` to create the preprocessor and processed data.
            *   Runs `train.py` to train the model and save it.
        3.  **Test:** Runs automated tests (e.g., unit tests for preprocessing functions, API tests).
        4.  **Build Docker Image:** Packages the FastAPI application and its dependencies (including the trained model and preprocessor) into a Docker image.
        5.  **Push to Registry:** Pushes the Docker image to a container registry (like GitLab Container Registry or Docker Hub).
        6.  **Deploy (Conceptual):** Includes steps to deploy the Docker image to a hosting environment like Kubernetes or Azure Container Apps.

8.  **Monitoring & Logging:**
    *   **Logging:** The Python scripts include logging to track their execution and potential errors.
    *   **Prometheus:** The FastAPI application is instrumented to expose metrics (like request counts, latency, error rates) that Prometheus can scrape.
    *   **Grafana:** A Grafana dashboard is provided to visualize the metrics collected by Prometheus, offering insights into the API's performance.

**Bonus Features:**

*   **Handling Data Imbalance:** The `train.py` script includes a basic way to handle class imbalance in the target variable (Churn) by using `scale_pos_weight` for XGBoost or `class_weight='balanced'` for Logistic Regression. More advanced techniques like SMOTE could also be implemented.
*   **Versioning:** MLflow is used for tracking experiments, parameters, metrics, and model artifacts, providing versioning for models and their associated preprocessing steps. DVC (Data Version Control) could be added for more robust data and model versioning.
*   **Auto-Retraining (Conceptual):** The pipeline is set up to be automated. Triggering retraining based on performance drops would involve setting up a monitoring system that checks model performance regularly and triggers the CI/CD pipeline if it degrades below a threshold.

**Deliverables (Found in the `customer-churn-prediction` directory):**

*   **`src/`**: Contains all Python source code for data ingestion, preprocessing, training, and the API.
*   **`data/`**: Holds the raw dataset (`Telco-Customer-Churn.csv`) and will contain processed data after running `preprocess.py`.
*   **`models/`**: Stores the saved preprocessor (`preprocessor.pkl`) and the trained model (`churn_model_xgb.pkl`).
*   **`notebooks/`**: For exploratory data analysis (`01_eda_and_prototyping.ipynb`).
*   **`monitoring/`**: Contains configurations for Prometheus and Grafana.
*   **`tests/`**: (To be expanded) Contains unit and integration tests.
*   **`Dockerfile`**: Instructions to build the FastAPI application Docker image.
*   **`docker-compose.yaml`**: Defines how to run the FastAPI app, Prometheus, and Grafana together locally using Docker Compose.
*   **`requirements.txt`**: Lists Python dependencies for the ML pipeline and API.
*   **`.gitlab-ci.yml`**: GitLab CI/CD pipeline configuration file.
*   **`main_pipeline.py`**: A script to run the data processing and model training steps locally.

---

## 2. Web Application (`churn-prediction-webapp/`)

This is a simple web application built with Flask that allows a user to input customer data and get a churn prediction by calling the deployed ML model API.

### Key Components:

1.  **`app.py` (Flask App):**
    *   Defines a web route (`/`) to display an HTML form.
    *   Defines another route (`/predict`) that:
        *   Receives customer data from the HTML form.
        *   Formats this data into the JSON structure expected by the churn prediction API.
        *   Sends a POST request to the churn prediction API (running on Kubernetes or Azure Container Apps).
        *   Receives the prediction response from the API.
        *   Displays the prediction (Churn: Yes/No, Probability) back to the user on the web page.
    *   Includes error handling for API communication issues.

2.  **`templates/index.html`:**
    *   An HTML file that creates a user-friendly form.
    *   The form includes input fields for all the features required by the churn model (gender, tenure, MonthlyCharges, etc.).
    *   Uses JavaScript to capture the form data, send it to the Flask backend (`/predict` endpoint), and display the result.

3.  **`static/style.css` (Optional):**
    *   Basic CSS for styling the HTML page to make it more presentable.

4.  **`requirements.txt`:**
    *   Lists Python dependencies for the web app (Flask, Requests).

5.  **`Dockerfile`:**
    *   Instructions to build a Docker image for the Flask web application. This allows it to be deployed consistently anywhere Docker is supported, including Kubernetes or Azure Container Apps.

### How it Works with the ML Pipeline:

1.  The **ML Pipeline** trains a churn prediction model and saves it (e.g., `churn_model_xgb.pkl`) along with its preprocessor (`preprocessor.pkl`).
2.  The FastAPI application (from the ML pipeline project) loads this model and preprocessor and exposes a `/predict` API endpoint. This API is then deployed (e.g., to Kubernetes or Azure Container Apps).
3.  The **Web Application** (`churn-prediction-webapp`) provides a user interface.
4.  When a user enters customer data and submits the form, the web app's backend (Flask) sends this data to the `/predict` endpoint of the deployed ML API.
5.  The ML API processes the data, makes a prediction, and returns the result to the web app.
6.  The web app then displays this prediction to the user.

### Running the Web App:

*   **Locally (for development):**
    *   Ensure the ML API is running (either locally or on Kubernetes/Azure Container Apps).
    *   Update the `CHURN_API_URL` in `app.py` or in a `.env` file to point to your running API.
    *   Navigate to the `churn-prediction-webapp` directory.
    *   Install dependencies: `pip install -r requirements.txt`
    *   Run the Flask app: `python app.py`
    *   Open your browser to `http://localhost:5001`.
*   **With Docker Compose (for local testing of both services):**
    *   Use the combined `docker-compose.yaml` in the main project root.
    *   Run `docker-compose up --build`.
    *   Access the web app at `http://localhost:5001`.
*   **On Azure Container Apps:**
    *   Build and push the Docker image for the web app.
    *   Deploy it as a new Container App, ensuring the `CHURN_API_URL` environment variable points to the internal URL of your backend API Container App.

---

This project structure and workflow provide a complete example of how to develop, deploy, and monitor a machine learning model, and then consume its predictions via a separate web application.

