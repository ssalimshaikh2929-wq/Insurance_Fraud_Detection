import os
import pandas as pd


# --------------------------------------------------
# Save model predictions into CSV file
# --------------------------------------------------

def save_predictions(
    input_data,
    predictions,
    probability,
    model_version,
    log_file
):

    # Create copy so original input data is not changed
    log_data = input_data.copy()

    # Add model prediction
    log_data["Prediction"] = predictions

    # Add fraud probability
    log_data["Probability"] = probability

    # Store model version
    log_data["Model_Version"] = model_version

    # Actual label is not known at prediction time
    log_data["Actual_Label"] = None


    # Check whether prediction log already exists
    file_exists = os.path.exists(log_file)


    # Save predictions into CSV
    log_data.to_csv(
        log_file,
        mode="a",                  # Append new predictions
        header=not file_exists,   # Header only for first save
        index=False
    )


# --------------------------------------------------
# Add actual labels later
# --------------------------------------------------

def add_actual_label(
    log_file,
    actual_labels
):

    # Read existing prediction logs
    df = pd.read_csv(log_file)

    # Safety check
    if len(actual_labels) != len(df):
        raise ValueError(
            "Number of actual labels must match "
            "number of logged predictions."
        )

    # Add actual fraud results
    df["Actual_Label"] = actual_labels


    # Save updated file
    df.to_csv(
        log_file,
        index=False
    )