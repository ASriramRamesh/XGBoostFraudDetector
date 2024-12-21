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
import matplotlib.pyplot as plt
import seaborn as sns

# --- Global Constants ---
RANDOM_STATE = 42
N_ITER = 10 # Number of random search iterations
CV_FOLDS = 5 # Cross-validation folds

# --- Database Connection (Placeholder) ---
# In a real-world scenario, replace with your actual database connection logic.
def fetch_data_from_db():
    """
    Placeholder for fetching data from a database.
    Returns a sample dataset.
    """
    data = """step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
283,CASH_IN,210329.84,C1159819632,3778062.79,3988392.64,C1218876138,1519266.6,1308936.76,0,0
132,CASH_OUT,215489.19,C1372369468,21518.0,0.0,C467105520,6345756.55,6794954.89,0,0
355,DEBIT,4431.05,C1059822709,20674.0,16242.95,C76588246,80876.56,85307.61,0,0
135,CASH_OUT,214026.2,C1464960643,46909.73,0.0,C1059379810,13467450.36,13681476.56,0,0
33,TRANSFER,535384.37,C1195440050,49783.0,0.0,C929909070,3933400.35,4468784.72,0,0
373,CASH_IN,17584.7,C1041108405,3159.0,19743.7,C1000256042,314692.29,297107.58,0,0
258,CASH_OUT,28101.66,C1477881534,0.0,0.0,C1638161161,1216149.99,1244251.65,0,0
352,CASH_IN,300124.81,C1210845813,55789.0,355913.81,C1553815217,625674.8,325549.99,0,0
43,CASH_IN,10380.54,C927829518,16773.0,27153.54,C1701425729,119106.92,108726.38,0,0
377,CASH_OUT,427243.45,C1726043234,127013.0,0.0,C1828157242,2943165.43,3370408.88,0,0
    """
    
    from io import StringIO
    df = pd.read_csv(StringIO(data))
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
def train_model(X_processed, y):
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
        X_processed, y, _ = preprocess_data(df)
        st.success("Data preprocessing completed!")
        st.write("Data Shape after Preprocessing:", X_processed.shape)
        

    elif phase == "Model Training and Evaluation":
        st.header("Model Training and Evaluation")
        
        df, _, _, _, _, _, _, _, _, _ = load_and_explore_data()
        X_processed, y, _ = preprocess_data(df)

        if st.button("Train Model"):
            with st.spinner('Training model...'):
                model, X_test, y_test = train_model(X_processed, y)
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
        X_processed, y, preprocessor = preprocess_data(df)

        if 'model' not in st.session_state:
             # Train the model if not already in session state
             with st.spinner('Training model...'):
                 model, X_test, y_test = train_model(X_processed, y)
                 st.session_state['model'] = model
             st.success("Model trained for the Prediction phase")
        else:
            model= st.session_state['model']

        # Input fields
        st.subheader("Enter transaction details for prediction:")
        input_data = {}

        # Select the transaction type
        transaction_type = st.selectbox("Select Transaction Type", options=df['type'].unique())
        input_data['type'] = transaction_type

        # Input fields for numerical columns
        for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            input_data[col] = st.number_input(f"Enter {col}", value=0.0)

        if st.button("Predict"):
           # Create a DataFrame from user inputs with correct column order and original df columns
            input_df = pd.DataFrame([input_data], columns=['type','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

            # Feature engineering same as in training
            input_df['balanceDiffOrg'] = input_df['oldbalanceOrg'] - input_df['newbalanceOrig']
            input_df['balanceDiffDest'] = input_df['oldbalanceDest'] - input_df['newbalanceDest']
            input_df['amountRatioOrg'] = input_df['amount'] / (input_df['oldbalanceOrg'] + 1e-8)
            input_df['amountRatioDest'] = input_df['amount'] / (input_df['oldbalanceDest'] + 1e-8)

           
            # Preprocess the data
            X_input_processed = preprocessor.transform(input_df)

            # Make Prediction
            prediction = model.predict(X_input_processed)
            
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("Fraudulent Transaction Detected!")
            else:
                st.success("Legitimate Transaction Detected.")

if __name__ == "__main__":
    main()