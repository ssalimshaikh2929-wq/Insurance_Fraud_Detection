from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,   # Scales numerical values around mean=0 and std=1
    MinMaxScaler,     # Scales numerical values usually between 0 and 1
    OneHotEncoder,    # Converts unordered categories into separate 0/1 columns
    OrdinalEncoder    # Converts ordered categories into numerical rankings
)


def create_preprocessing_pipeline():

    # -------------------------------------------------
    # 1. StandardScaler columns
    # Used for continuous numerical features.
    # StandardScaler → mean = 0, standard deviation = 1
    # -------------------------------------------------

    standard_scaler_columns = [
        "months_as_customer",
        "age",
        "policy_annual_premium",
        "umbrella_limit",
        "capital-gains",
        "capital-loss",
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "policy_age_days"
    ]


    # -------------------------------------------------
    # 2. MinMaxScaler columns
    # Scales numerical values usually between 0 and 1.
    # Here we use it for small-range/bounded features.
    # -------------------------------------------------

    minmax_scaler_columns = [
        "policy_deductable",
        "incident_hour_of_the_day",
        "number_of_vehicles_involved",
        "bodily_injuries",
        "witnesses",
        "auto_year",
        "incident_month",
        "incident_day_of_week"
    ]


    # -------------------------------------------------
    # 3. OneHotEncoder columns
    # Used for NOMINAL categorical features.
    # Nominal = categories have NO meaningful order.
    #
    # Example:
    # Male / Female
    # Ohio / Indiana / Illinois
    #
    # Converts categories into separate 0/1 columns.
    # -------------------------------------------------

    onehot_columns = [
        "policy_state",
        "policy_csl",
        "insured_sex",
        "insured_education_level",
        "insured_occupation",
        "insured_hobbies",
        "insured_relationship",
        "incident_type",
        "collision_type",
        "authorities_contacted",
        "incident_state",
        "incident_city",
        "property_damage",
        "police_report_available",
        "auto_make",
        "auto_model"
    ]


    # -------------------------------------------------
    # 4. OrdinalEncoder columns
    # Used when categories HAVE a meaningful order.
    #
    # Example:
    # Trivial < Minor < Major < Total Loss
    # -------------------------------------------------

    ordinal_columns = [
        "incident_severity"
    ]


    # =================================================
    # STANDARD SCALER PIPELINE
    # =================================================

    standard_pipeline = Pipeline([

        # Fill missing numerical values using median.
        # Median is less affected by outliers than mean.
        ("imputer",SimpleImputer(strategy="median")),

        # Standardize numerical values.
        # Output is centered around mean=0 and std=1.
        ("scaler",StandardScaler())
    ])


    # =================================================
    # MINMAX SCALER PIPELINE
    # =================================================

    minmax_pipeline = Pipeline([

        # Fill missing numerical values with median.
        ("imputer",SimpleImputer(strategy="median")),

        # Scale values usually between 0 and 1.
        ("scaler",MinMaxScaler())
    ])


    # =================================================
    # ONE HOT ENCODING PIPELINE
    # =================================================

    onehot_pipeline = Pipeline([

        # Fill missing categorical values
        # using the most common category.
        ("imputer",SimpleImputer(strategy="most_frequent")),

        # Convert unordered categories into 0/1 columns.
        # handle_unknown="ignore" prevents errors if
        # test/new data contains an unseen category.
        ("encoder",OneHotEncoder(handle_unknown="ignore"))
    ])


    # =================================================
    # ORDINAL ENCODING PIPELINE
    # =================================================

    ordinal_pipeline = Pipeline([

        # Fill missing values with most frequent category.
        ("imputer",SimpleImputer(strategy="most_frequent")),

        # Convert ordered categories into numbers.
        ("encoder",
            OrdinalEncoder(
                categories=[[
                    "Trivial Damage",
                    "Minor Damage",
                    "Major Damage",
                    "Total Loss"
                ]],

                # If new/unseen category appears,
                # encode it as -1 instead of giving an error.
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ])


    # =================================================
    # COMBINE ALL PREPROCESSING
    # =================================================

    # ColumnTransformer applies different preprocessing
    # pipelines to different groups of columns.

    preprocessing_pipeline = ColumnTransformer([

        # Continuous numerical → StandardScaler
        ("standard",standard_pipeline,standard_scaler_columns),

        # Bounded numerical → MinMaxScaler
        ("minmax",minmax_pipeline,minmax_scaler_columns),

        # Unordered categorical → OneHotEncoder
        ("onehot",onehot_pipeline,onehot_columns),

        # Ordered categorical → OrdinalEncoder
        ("ordinal",ordinal_pipeline,ordinal_columns)
    ])

    return preprocessing_pipeline