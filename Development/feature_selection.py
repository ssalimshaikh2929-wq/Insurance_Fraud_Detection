import pandas as pd


def select_features(df):

    columns_to_drop = [
        "policy_number",
        "insured_zip",
        "incident_location",
        "fraud_reported"
    ]

    # Features
    X = df.drop(
        columns=columns_to_drop
    
    )

    # Target
    if "fraud_reported" in df.columns:
        y = df["fraud_reported"].map({
            "Y": 1,
            "N": 0
        })
    
        return X, y

    return X

 