import pandas as pd


def create_features(df):

    # Create a copy
    df = df.copy()

    # Convert columns to datetime
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])

    df["incident_date"] = pd.to_datetime(df["incident_date"])

    # Policy age at the time of incident
    df["policy_age_days"] = (df["incident_date"] - df["policy_bind_date"]
    ).dt.days

    # Incident month
    df["incident_month"] = (df["incident_date"].dt.month)

    # Incident day of week
    df["incident_day_of_week"] = (df["incident_date"].dt.dayofweek)

    # Remove original date columns
    df.drop(
        columns=[
            "policy_bind_date",
            "incident_date"
        ],
        inplace=True
    )

    return df