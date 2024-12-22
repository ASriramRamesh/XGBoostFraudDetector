# Create virtual environment with python. I have python 3.12.4 installed
python -m venv venv_fraud_detection

# After creating virtual environment run the requirements.txt
pip install -r requirements. txt

# Activate virtual environment first in the terminal
venv_fraud_detection\Scripts\activate

# After installing the dependencies create the requirements.txt file
pip freeze > requirements. txt 

# Run streamlit command in the terminal
streamlit run credit_card_app.py

# Streamlit Credit Card Fraud Detection Application

## 🌟 Overview
This application is designed to detect fraudulent credit card transactions using an **XGBoost** machine learning model. It provides an interactive interface for:
- Data exploration
- Model training and evaluation
- Fraud prediction

---

## 🚀 Core Features

### 1. **Data Loading and Exploration**
- **`fetch_data_from_db()`**: Simulates fetching transaction data from a database.
- **`load_and_explore_data()`**:
  - Loads and analyzes data, calculating descriptive statistics, missing values, and target variable distribution.
  - Generates interactive visualizations for:
    - **Transaction Data**: Histograms, box plots, and bar charts.
    - **Time Features**: Distribution of step days and weeks for fraud vs. non-fraud transactions.
    - **Name Features**: Frequency of the first letter of origin/destination names by fraud status.
    - **Correlations**: Heatmaps for numerical and categorical features.
  - Caches results for faster performance.

### 2. **Data Preprocessing**
- **`preprocess_data()`**:
  - **Feature Engineering**:
    - Computes balance differences and transaction amount ratios.
    - Creates time-based features (e.g., `step_days`, `step_weeks`).
    - Extracts initials of origin and destination names.
  - Handles outliers by capping extreme values.
  - Applies transformations using a `ColumnTransformer`:
    - Numerical: Imputation (median) and scaling (QuantileTransformer).
    - Categorical: One-hot encoding.
  - Returns processed data, target variable, and the preprocessor.

### 3. **Model Training**
- **`train_model()`**:
  - Splits data (80/20, stratified sampling).
  - Optimizes **XGBoost** parameters using RandomizedSearchCV.
  - Trains the best model and saves it along with the preprocessor.

### 4. **Model Evaluation**
- **`evaluate_model()`**:
  - Predicts outcomes on test data.
  - Calculates metrics: accuracy, precision, recall, F1-score, and confusion matrix.

### 5. **Interactive Streamlit UI**
The application is organized into phases for user convenience:

#### **Phase 1: Data Exploration**
- Explore data distributions, missing values, and feature behaviors through visualizations.

#### **Phase 2: Data Preprocessing**
- Preprocess data and view the shape of processed datasets.

#### **Phase 3: Model Training and Evaluation**
- Train the model and evaluate its performance.
- Display evaluation metrics, confusion matrix, and feature importance.

#### **Phase 4: Predictions**
- Upload a CSV file to predict fraudulent transactions.
- Automatically preprocess data, make predictions, and evaluate results.

---

## 💡 Key Highlights
- **Modular Design**: Reusable functions for each task (loading, preprocessing, training).
- **Interactive Visualizations**: Powered by Plotly for dynamic insights.
- **Model Persistence**: Save trained models and preprocessors to avoid repeated training.
- **Outlier Handling**: Caps extreme values to prevent adverse effects on modeling.
- **Feature Engineering**: Enhances predictive power with custom features.
- **Data Transformation**: Scales numerical data and encodes categorical features effectively.
- **Feature Selection**: Removes redundant features to improve generalization.

---

## 🛠️ Workflow
1. **Data Exploration**
   - Visualize and analyze data distributions, correlations, and feature relationships.
2. **Data Preprocessing**
   - Transform and clean data for better model performance.
3. **Model Training**
   - Train an XGBoost model with hyperparameter tuning.
   - Save the trained model and preprocessor.
4. **Model Evaluation**
   - Assess the model's performance with metrics and visualizations.
5. **Prediction**
   - Predict fraud on new data and evaluate the model against provided ground truths.

---

## 📈 Key Features of the Model
- **Algorithm**: XGBoost with hyperparameter tuning.
- **Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Visualization**: Feature importance, confusion matrix.

---

## 🖥️ How to Use
1. **Run the Application**:
   ```
   streamlit run app.py
   ```
2. **Explore Data**: Navigate through exploration phases for insights.
3. **Train Model**: Train and evaluate the model in real time.
4. **Predict**: Upload a CSV to detect fraud and view predictions.

---

## 🔑 Key Considerations
- Ensure proper feature engineering and preprocessing for accurate predictions.
- Save models and preprocessors to maintain consistent results.
- Use uploaded CSV files with correct formats for prediction.

---

## 🤝 Contributing
Contributions are welcome! Please fork this repository and create a pull request to suggest improvements or report issues.

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

---


