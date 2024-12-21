import pandas as pd
import numpy as np
import os

def create_bifurcated_data(data_path, output_path):
    """
    Loads the fraud_detect.csv data, splits it into six dataframes, and saves them to the specified output directory.

    Args:
        data_path (str): The path to the input data.
        output_path (str): The path to the output directory to save the splitted dataframes.
    """

    # Load the fraud_detect.csv file
    try:
      df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return

    # Check if isFraud column exist
    if "isFraud" not in df.columns:
        print(f"The csv {data_path} does not contain a isFraud column")
        return

    # Filter fraud and non-fraud cases
    fraud_cases = df[df["isFraud"] == 1]
    non_fraud_cases = df[df["isFraud"] == 0]

    # Number of prediction dataframes to create
    num_prediction_dfs = 5
    records_per_prediction_df = 10
    num_fraud_records = 2

    # Create prediction dataframes
    for i in range(num_prediction_dfs):
       # Select records. First get 2 fraud records, followed by remaining non_fraud records
        df_fraud_sample = fraud_cases.sample(n=num_fraud_records, replace=False)
        df_non_fraud_sample = non_fraud_cases.sample(n=records_per_prediction_df - num_fraud_records, replace=False)

        df_predict = pd.concat([df_fraud_sample, df_non_fraud_sample])

        # Remove sampled rows from main dataframes, so that we do not resample
        fraud_cases = fraud_cases.drop(df_fraud_sample.index)
        non_fraud_cases = non_fraud_cases.drop(df_non_fraud_sample.index)

        # Shuffle the data
        df_predict = df_predict.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True)

        # Save the prediction dataframe
        file_path = os.path.join(output_path, f"predict_{i+1}.csv")
        df_predict.to_csv(file_path, index=False)

    # Create test/train dataframe from the remaining data
    df_test_train = pd.concat([fraud_cases, non_fraud_cases])
    df_test_train = df_test_train.sample(frac=1, random_state = RANDOM_STATE).reset_index(drop=True)
    test_train_path = os.path.join(output_path, "test_train.csv")
    df_test_train.to_csv(test_train_path, index = False)

    print(f"Splitted dataframes and saved in {output_path}")


if __name__ == "__main__":
    # Define folder paths
    RANDOM_STATE = 42
    data_dir = "data"  # Path for the folder that holds the data
    fraud_detect_file = "fraud_detect.csv"
    output_dir = "data" # Folder where the new data will be saved

    # Get current working directory
    current_dir = os.getcwd()
    
    # Create folders if they don't exist
    data_path = os.path.join(current_dir, data_dir, fraud_detect_file)
    output_path = os.path.join(current_dir, output_dir)

    if not os.path.exists(output_path):
         os.makedirs(output_path)
         print(f"Created folder {output_path} in current working directory")

    create_bifurcated_data(data_path, output_path)