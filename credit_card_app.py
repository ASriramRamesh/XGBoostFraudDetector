import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.figure_factory as ff
import os
import joblib  # For saving and loading models and preprocessors

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Global Constants ---
RANDOM_STATE = 42
N_ITER = 10  # Number of random search iterations
CV_FOLDS = 5  # Cross-validation folds
MODEL_FOLDER = "models"
MODEL_FILE = "trained_model.pkl"
PREPROCESSOR_FOLDER = "preprocessors"
PREPROCESSOR_FILE = "preprocessor.pkl"
OUTLIER_THRESHOLD = 0.01  # Threshold for outlier removal

class SklearnCompatibleXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __sklearn_tags__(self):
        return {"estimator_type": "classifier", "requires_positive_y": False}

# --- Database Connection (Placeholder) ---
# In a real-world scenario, replace with your actual database connection logic.
def fetch_data_from_db():
    """
    Placeholder for fetching data from a database.
    Returns a sample dataset.
    """
    file_path = "data/test_train.csv"

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

    # --- Visualizations - Section 1 ---
    hist_amount = px.histogram(df, x="amount", title="Transaction Amount Distribution")
    box_amount_fraud = px.box(df, x="isFraud", y="amount", title="Amount Distribution by Fraud")
    bar_type = px.bar(df, x="type", title="Transaction Type Frequency")

    # --- Visualizations - Section 2 ---
    df['step_days'] = df['step'] / 24
    df['step_weeks'] = df['step'] / (24 * 7)
    box_step_days_fraud = px.box(df, x="isFraud", y="step_days", title="Step Days Distribution by Fraud")
    box_step_weeks_fraud = px.box(df, x="isFraud", y="step_weeks", title="Step Weeks Distribution by Fraud")

    # --- Visualizations - Section 3 ---
    df['name_orig_initial'] = df['nameOrig'].str[0]
    df['name_dest_initial'] = df['nameDest'].str[0]
    bar_name_orig_fraud = px.bar(df, x="name_orig_initial", color="isFraud", title="Name Orig Initial by Fraud")
    bar_name_dest_fraud = px.bar(df, x="name_dest_initial", color="isFraud", title="Name Dest Initial by Fraud")

    cat_corr = df.select_dtypes(include='object').apply(lambda x: x.astype('category').cat.codes)
    cat_corr['isFraud'] = df['isFraud']
    cat_corr_matrix = cat_corr.corr()

    cat_corr_heatmap_fig = ff.create_annotated_heatmap(z=cat_corr_matrix.values,
                                              x=list(cat_corr_matrix.columns),
                                              y=list(cat_corr_matrix.index),
                                              colorscale='Viridis',
                                              showscale=False
                                             )

    correlation_matrix = df.select_dtypes(include=np.number).corr()

    heatmap_fig = ff.create_annotated_heatmap(z=correlation_matrix.values,
                                              x=list(correlation_matrix.columns),
                                              y=list(correlation_matrix.index),
                                              colorscale='Viridis',
                                              showscale=False
                                              )

    return df, summary_stats, missing_values, fraud_count, fraud_percentage, \
        hist_amount, box_amount_fraud, bar_type, \
        box_step_days_fraud, box_step_weeks_fraud, \
        bar_name_orig_fraud, bar_name_dest_fraud, cat_corr_heatmap_fig, heatmap_fig, cat_corr_matrix, correlation_matrix

# --- Data Preprocessing ---
def preprocess_data(df):
    """Preprocesses the data using a pipeline."""

    # Define numerical and categorical features
    numerical_features = ['amount', 'balanceDiffOrg', 'balanceDiffDest', 'amountRatioOrg', 'amountRatioDest',
                          'step_days', 'step_weeks']
    categorical_features = ['type', 'name_orig_initial', 'name_dest_initial']

    # Feature Engineering
    df['balanceDiffOrg'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['amountRatioOrg'] = df['amount'] / (df['oldbalanceOrg'] + 1e-8)
    df['amountRatioDest'] = df['amount'] / (df['oldbalanceDest'] + 1e-8)
    df['step_days'] = df['step'] / 24
    df['step_weeks'] = df['step'] / (24 * 7)

    df['name_orig_initial'] = df['nameOrig'].str[0].fillna('N/A')
    df['name_dest_initial'] = df['nameDest'].str[0].fillna('N/A')

    # --- Remove Outliers ---
    for col in ['amount', 'balanceDiffOrg', 'balanceDiffDest', 'amountRatioOrg', 'amountRatioDest']:
        lower_quantile = df[col].quantile(OUTLIER_THRESHOLD)
        upper_quantile = df[col].quantile(1 - OUTLIER_THRESHOLD)
        df[col] = np.clip(df[col], lower_quantile, upper_quantile)

    # --- Drop redundant Features ---
    df = df.drop(columns=['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud','step','nameOrig','nameDest'])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', QuantileTransformer(output_distribution='normal'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor  # Return the preprocessor in order to use in the inference section

# --- Model Training ---
def train_model(X_processed, y, preprocessor):
    """Trains an XGBoost model with hyperparameter tuning using RandomizedSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

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

    # Initialize XGBoost Classifier with explicit enable_categorical=False
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        enable_categorical=False,  # Explicitly disable categorical feature support
        use_label_encoder=False   # Disable label encoder warning
    )

    # # Use SklearnCompatibleXGBClassifier instead of XGBClassifier
    # xgb_model = SklearnCompatibleXGBClassifier(
    #     eval_metric='logloss',
    #     random_state=RANDOM_STATE,
    #     enable_categorical=False,
    #     use_label_encoder=False
    # )

    # Create custom StratifiedKFold object
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Setup RandomizedSearchCV with explicit cv parameter
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        error_score='raise'
    )

    # Fit the RandomizedSearchCV
    random_search.fit(X_train_balanced, y_train_balanced)

    # Get the best model
    best_model = random_search.best_estimator_

    # Save the model and preprocessor
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    if not os.path.exists(PREPROCESSOR_FOLDER):
        os.makedirs(PREPROCESSOR_FOLDER)

    model_path = os.path.join(MODEL_FOLDER, MODEL_FILE)
    preprocessor_path = os.path.join(PREPROCESSOR_FOLDER, PREPROCESSOR_FILE)

    joblib.dump(best_model, model_path)
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
                             ("Data Exploration Phase 1",
                              "Data Exploration Phase 2",
                              "Data Exploration Phase 3",
                              "Data Preprocessing",
                              "Model Training and Evaluation",
                              "Make Predictions"))

    if phase == "Data Exploration Phase 1":
        st.header("Data Exploration - Transaction Data")
        df, summary_stats, missing_values, fraud_count, fraud_percentage, \
            hist_amount, box_amount_fraud, bar_type, \
             box_step_days_fraud, box_step_weeks_fraud, \
             bar_name_orig_fraud, bar_name_dest_fraud, cat_corr_heatmap_fig, heatmap_fig, _, _ = load_and_explore_data()

        st.subheader("Data Summary")
        st.dataframe(summary_stats)
        st.subheader("Missing Values")
        st.write(missing_values)
        st.subheader("Target Variable Analysis")
        st.write(f"Fraud Count:\n{fraud_count}")
        st.write(f"Fraud Percentage:\n{fraud_percentage}")

        st.plotly_chart(hist_amount)
        st.plotly_chart(box_amount_fraud)
        st.plotly_chart(bar_type)

    elif phase == "Data Exploration Phase 2":
        st.header("Data Exploration - Time Features")

        df, summary_stats, missing_values, fraud_count, fraud_percentage, \
            hist_amount, box_amount_fraud, bar_type, \
            box_step_days_fraud, box_step_weeks_fraud, \
            bar_name_orig_fraud, bar_name_dest_fraud, cat_corr_heatmap_fig, heatmap_fig, _, _ = load_and_explore_data()

        st.plotly_chart(box_step_days_fraud)
        st.plotly_chart(box_step_weeks_fraud)

    elif phase == "Data Exploration Phase 3":
        st.header("Data Exploration - Name Initial and Categorical Correlations")

        df, summary_stats, missing_values, fraud_count, fraud_percentage, \
            hist_amount, box_amount_fraud, bar_type, \
            box_step_days_fraud, box_step_weeks_fraud, \
            bar_name_orig_fraud, bar_name_dest_fraud, _, _, cat_corr_matrix, correlation_matrix = load_and_explore_data()

        st.plotly_chart(bar_name_orig_fraud)
        st.plotly_chart(bar_name_dest_fraud)

        # --- Modified cat_corr_heatmap_fig ---
        categorical_correlation_values = cat_corr_matrix.values
        categorical_column_names = list(cat_corr_matrix.columns)
        categorical_row_names = list(cat_corr_matrix.index)

        # Create annotations
        annotations_cat_text = np.empty_like(categorical_correlation_values, dtype=str)
        for i, row in enumerate(categorical_correlation_values):
            for j, val in enumerate(row):
                if abs(val) >= 0.5:  # Adjust this threshold as needed
                    annotations_cat_text[i, j] = f'{val:.1f}'
                else:
                    annotations_cat_text[i, j] = ''

        cat_corr_heatmap_fig = ff.create_annotated_heatmap(z=categorical_correlation_values,
                                            x=categorical_column_names,
                                            y=categorical_row_names,
                                            colorscale='Viridis',
                                            showscale=False,
                                            annotation_text=annotations_cat_text
                                            )
        cat_corr_heatmap_fig.update_layout(width=800, height=800) # Increase heatmap size

        st.plotly_chart(cat_corr_heatmap_fig)

        # --- Modified heatmap_fig ---
        numerical_correlation_values = correlation_matrix.values
        numerical_column_names = list(correlation_matrix.columns)
        numerical_row_names = list(correlation_matrix.index)

        # Create annotations
        annotations_num_text = np.empty_like(numerical_correlation_values, dtype=str)
        for i, row in enumerate(numerical_correlation_values):
            for j, val in enumerate(row):
                if abs(val) >= 0.5:  # Adjust this threshold as needed
                    annotations_num_text[i, j] = f'{val:.1f}'
                else:
                    annotations_num_text[i, j] = ''

        heatmap_fig = ff.create_annotated_heatmap(z=numerical_correlation_values,
                                            x=numerical_column_names,
                                            y=numerical_row_names,
                                            colorscale='Viridis',
                                            showscale=False,
                                            annotation_text = annotations_num_text
                                            )
        heatmap_fig.update_layout(width=800, height=800) # Increase heatmap size

        st.plotly_chart(heatmap_fig)
    elif phase == "Data Preprocessing":

        st.header("Data Preprocessing")
        df, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        X_processed, y, preprocessor = preprocess_data(df)
        st.success("Data preprocessing completed!")
        st.write("Data Shape after Preprocessing:", X_processed.shape)

    elif phase == "Model Training and Evaluation":
        st.header("Model Training and Evaluation")

        df, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        X_processed, y, preprocessor = preprocess_data(df)

        if st.button("Train Model"):
            with st.spinner('Training model...'):
                model, X_test, y_test = train_model(X_processed, y, preprocessor)
                # Feature Importance
                feature_importance = pd.Series(model.feature_importances_, index = preprocessor.get_feature_names_out()).sort_values(ascending = False)
                fig = px.bar(x = feature_importance.index, y = feature_importance, title = 'Feature Importance')
                st.plotly_chart(fig)

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

        df, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
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
                input_df_features['step_days'] = input_df_features['step'] / 24
                input_df_features['step_weeks'] = input_df_features['step'] / (24 * 7)

                input_df_features['name_orig_initial'] = input_df_features['nameOrig'].str[0].fillna('N/A')
                input_df_features['name_dest_initial'] = input_df_features['nameDest'].str[0].fillna('N/A')

                # Drop redundant columns
                input_df_features = input_df_features.drop(columns=['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest','step','nameOrig','nameDest','isFlaggedFraud'])

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
