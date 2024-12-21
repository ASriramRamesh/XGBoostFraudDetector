import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.figure_factory as ff
import os
import joblib # For saving and loading models and preprocessors

# --- Global Constants ---
RANDOM_STATE = 42
N_ITER = 10 # Number of random search iterations
CV_FOLDS = 5 # Cross-validation folds
MODEL_FOLDER = "models"
MODEL_FILE = "trained_model.pkl"
PREPROCESSOR_FOLDER = "preprocessors"
PREPROCESSOR_FILE = "preprocessor.pkl"

# --- Database Connection (Placeholder) ---
# In a real-world scenario, replace with your actual database connection logic.
def fetch_data_from_db():
    """
    Placeholder for fetching data from a database.
    Returns a sample dataset.
    """
    file_path = "data/fraud_detect.csv"

    df = pd.read_csv(file_path)
    return df

# --- Data Loading and EDA ---
@st.cache_data
def load_and_explore_data():
    """Loads data, performs basic EDA, and returns relevant information."""
    df = fetch_data_from_db()

    # Data Summary
    summary_stats = df.describe()
    missing_values = df.isnull().sum()
    
    # Target Variable Analysis
    fraud_count = df['isFraud'].value_counts()
    fraud_percentage = df['isFraud'].value_counts(normalize=True) * 100

    # Visualizations (using Plotly for interactive plots)
    hist_amount = px.histogram(df, x="amount", title="Transaction Amount Distribution")
    box_amount_fraud = px.box(df, x="isFraud", y="amount", title="Amount Distribution by Fraud")
    bar_type = px.bar(df, x="type", title="Transaction Type Frequency")
    scatter_amount_balance = px.scatter(df, x="oldbalanceOrg", y="amount", color="isFraud", title="Amount vs. Old Balance")
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    
    heatmap_fig = ff.create_annotated_heatmap(z=correlation_matrix.values,
                                              x=list(correlation_matrix.columns),
                                              y=list(correlation_matrix.index),
                                              colorscale='Viridis',
                                             )

    return df, summary_stats, missing_values, fraud_count, fraud_percentage, hist_amount, box_amount_fraud, bar_type, scatter_amount_balance, heatmap_fig

# --- Data Preprocessing ---
def preprocess_data(df):
    """Preprocesses the data using a pipeline."""
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    categorical_features = ['type']

    # Feature Engineering
    df['balanceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['amountRatioOrg'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)  # Add small constant to avoid division by zero
    df['amountRatioDest'] = df['amount'] / (df['oldbalanceDest'] + 1e-8)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features + ['balanceDiffOrg', 'balanceDiffDest', 'amountRatioOrg', 'amountRatioDest']),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X = df.drop(columns=['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'])
    y = df['isFraud']
    
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor  #Return the preprocessor in order to use in the inference section

# --- Model Training ---
def train_model(X_processed, y, preprocessor):
    """Trains an XGBoost model with hyperparameter tuning using RandomizedSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Define the hyperparameter grid for randomized search
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 3, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    # Initialize XGBoost Classifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_dist, 
        n_iter=N_ITER, 
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1',
        n_jobs=-1,  # Use all available cores
        random_state=RANDOM_STATE
    )

    # Fit the RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # --- Save the model to the models folder ---
    if not os.path.exists(MODEL_FOLDER):
         os.makedirs(MODEL_FOLDER)

    model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
    joblib.dump(best_model, model_path) # Save the model and the preprocessor to the same folder
    
    # --- Save the preprocessor to the preprocessors folder ---
    if not os.path.exists(PREPROCESSOR_FOLDER):
        os.makedirs(PREPROCESSOR_FOLDER)

    preprocessor_path = os.path.join(PREPROCESSOR_FOLDER, PREPROCESSOR_FILE)
    joblib.dump(preprocessor, preprocessor_path)
    
    return best_model, X_test, y_test

# --- Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and returns the metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix

# --- Streamlit App ---
def main():
    st.title("Credit Card Fraud Detection System")
    
    # --- Sidebar Menu ---
    st.sidebar.title("Project Phases")
    phase = st.sidebar.radio("Select Phase", 
                             ("Data Exploration", 
                              "Data Preprocessing", 
                              "Model Training and Evaluation",
                              "Make Predictions"))

    if phase == "Data Exploration":
        st.header("Data Exploration")
        df, summary_stats, missing_values, fraud_count, fraud_percentage, hist_amount, box_amount_fraud, bar_type, scatter_amount_balance, heatmap_fig = load_and_explore_data()
        st.subheader("Data Summary")
        st.dataframe(summary_stats)
        st.subheader("Missing Values")
        st.write(missing_values)
        st.subheader("Target Variable Analysis")
        st.write(f"Fraud Count:\n{fraud_count}")
        st.write(f"Fraud Percentage:\n{fraud_percentage}")
        st.subheader("Visualizations")
        st.plotly_chart(hist_amount)
        st.plotly_chart(box_amount_fraud)
        st.plotly_chart(bar_type)
        st.plotly_chart(scatter_amount_balance)
        st.plotly_chart(heatmap_fig)

    elif phase == "Data Preprocessing":
       
        st.header("Data Preprocessing")
        df, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        X_processed, y, preprocessor = preprocess_data(df)
        st.success("Data preprocessing completed!")
        st.write("Data Shape after Preprocessing:", X_processed.shape)
        

    elif phase == "Model Training and Evaluation":
        st.header("Model Training and Evaluation")
        
        df, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        X_processed, y, preprocessor = preprocess_data(df)

        if st.button("Train Model"):
            with st.spinner('Training model...'):
                model, X_test, y_test = train_model(X_processed, y, preprocessor)
                accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)

                st.success("Model training completed!")

                st.subheader("Model Evaluation Metrics")
                st.write(f"Accuracy: {accuracy:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
                st.write(f"F1 Score: {f1:.4f}")
                st.subheader("Confusion Matrix")
                st.dataframe(conf_matrix)

    elif phase == "Make Predictions":
        st.header("Make Predictions")
        
        df, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        _, _, preprocessor = preprocess_data(df)

        # Define model and preprocessor paths
        model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
        preprocessor_path = os.path.join(PREPROCESSOR_FOLDER, PREPROCESSOR_FILE)


        # Check if saved model and preprocessor exist, and load them if present
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
             with st.spinner('Loading saved model and preprocessor...'):
                model = joblib.load(model_path)
                preprocessor = joblib.load(preprocessor_path)
                st.success("Saved model and preprocessor loaded successfully!")
        else:
            # Train the model if not already in session state
            with st.spinner('Training model...'):
                 X_processed, y, preprocessor = preprocess_data(df)
                 model, X_test, y_test = train_model(X_processed, y, preprocessor)

            st.success("Model trained for the Prediction phase")


        # Upload CSV file
        uploaded_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)

                 # Check if 'isFraud' column exists
                if 'isFraud' not in input_df.columns:
                    st.error("The CSV file must contain an 'isFraud' column.")
                    return

                # Get target column from input_df and remove it from input features before pre-processing
                y_true = input_df['isFraud']
                input_df_features = input_df.drop(columns=['isFraud'])


                # Feature engineering same as training
                input_df_features['balanceDiffOrg'] = input_df_features['oldbalanceOrg'] - input_df_features['newbalanceOrig']
                input_df_features['balanceDiffDest'] = input_df_features['oldbalanceDest'] - input_df_features['newbalanceDest']
                input_df_features['amountRatioOrg'] = input_df_features['amount'] / (input_df_features['oldbalanceOrg'] + 1e-8)
                input_df_features['amountRatioDest'] = input_df_features['amount'] / (input_df_features['oldbalanceDest'] + 1e-8)

                # Preprocess data and predict
                X_input_processed = preprocessor.transform(input_df_features)
                y_pred = model.predict(X_input_processed)

                # Evaluate model performance
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                conf_matrix = confusion_matrix(y_true, y_pred)

                 # Display Results
                st.subheader("Model Evaluation Metrics")
                st.write(f"Accuracy: {accuracy:.4f}")
                st.write(f"Precision: {precision:.4f}")
                st.write(f"Recall: {recall:.4f}")
                st.write(f"F1 Score: {f1:.4f}")
                st.subheader("Confusion Matrix")
                st.dataframe(conf_matrix)

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()