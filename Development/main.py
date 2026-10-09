# ============================================================
# main.py
# INSURANCE FRAUD DETECTION PROJECT
# ============================================================

import numpy as np
import pandas as pd
import joblib
import mlflow


# ============================================================
# IMPORT FUNCTIONS FROM PROJECT FILES
# ============================================================

from load_data import load_data
from feature_enggineering import create_features
from feature_selection import select_features
from train_test_split import split_data
from data_preprocessing import create_preprocessing_pipeline
from handle_balancing import handle_imbalance
from model_training import train_models
from model_evaluation import evaluate_models
from mlflow_tracking import (
    start_experiment,
    start_run,
    log_parameters,
    log_metrics,
    log_tags,
    log_model
)


# Start MLflow experiment
start_experiment("Insurance Fraud Detection")


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "Data/insurance_claims.csv"

df = load_data(file_path)

print("\n======================================")
print("1. DATA LOADED")
print("======================================")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Creates:
# policy_age_days
# incident_month
# incident_day_of_week
#
# Also removes:
# policy_bind_date
# incident_date

df = create_features(df)

print("\n======================================")
print("2. FEATURE ENGINEERING COMPLETED")
print("======================================")

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

# X = independent features
# y = fraud target
#
# fraud_reported:
# Y -> 1
# N -> 0

X, y = select_features(df)

print("\n======================================")
print("3. FEATURE SELECTION COMPLETED")
print("======================================")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = split_data(
    X,
    y
)
print("\n======================================")
print("4. TRAIN TEST SPLIT COMPLETED")
print("======================================")



# ============================================================
# 5. CONVERT "?" INTO MISSING VALUES
# ============================================================

# Dataset contains "?" in columns such as:
#
# collision_type
# property_damage
# police_report_available
#
# Convert them into np.nan.
#
# SimpleImputer inside preprocessing will handle these missing values.

X_train = X_train.replace(
    "?",
    np.nan
)

X_test = X_test.replace(
    "?",
    np.nan
)

print("\n======================================")
print("5. '?' VALUES CONVERTED TO NaN")
print("======================================")

# ============================================================
# 6. CREATE PREPROCESSING PIPELINE
# ============================================================

preprocessing_pipeline = create_preprocessing_pipeline()


# ============================================================
# 7. FIT PREPROCESSOR ON TRAINING DATA
# ============================================================

# fit_transform:
#
# 1. Learns imputation values
# 2. Learns scaling values
# 3. Learns OneHot categories
# 4. Transforms training data

X_train_processed = preprocessing_pipeline.fit_transform(
    X_train
)

# ============================================================
# 8. TRANSFORM TEST DATA
# ============================================================

# IMPORTANT:
# Do NOT use fit_transform on test data.
# We use the rules learned from training data.

X_test_processed = preprocessing_pipeline.transform(
    X_test
)


print("\n======================================")
print("6. PREPROCESSING COMPLETED")
print("======================================")

print("Processed X_train:",X_train_processed.shape)
print("Processed X_test:",X_test_processed.shape )



# ============================================================
# 9. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n======================================")
print("7. BEFORE BALANCING")
print("======================================")

print(y_train.value_counts())


# ============================================================
# 10. HANDLE CLASS IMBALANCE
# ============================================================

# We will test all three balancing methods:
# 1. SMOTE
# 2. Random Over Sampling
# 3. Random Under Sampling

methods = [
    "smote",
    "random_over",
    "random_under"
]


# This list will store evaluation results
# from all three balancing methods.
all_results = []


# This dictionary will store all trained models.
#
# Example key:
# ("smote", "Random Forest")
#
# This allows us to retrieve the correct model later.
all_trained_models = {}



# ============================================================
# LOOP THROUGH ALL BALANCING METHODS
# ============================================================

for method in methods:

    X_train_resampled, y_train_resampled = handle_imbalance(
        X_train_processed,
        y_train,
        method=method
    )


    print("\nClass Distribution Before Resampling:")

    print(
        y_train.value_counts()
    )

    print(
        y_train_resampled.value_counts()
    )


    # ========================================================
    # 11. TRAIN MODELS
    # ========================================================

    # train_models() will train:
    #
    # Logistic Regression
    # Random Forest
    # Support Vector Machine
    # Decision Tree

    trained_models = train_models(
        X_train_resampled,
        y_train_resampled
    )


    print("\n======================================")
    print("MODEL TRAINING COMPLETED")
    print("======================================")



    # ========================================================
    # 12. MODEL EVALUATION
    # ========================================================

    # IMPORTANT:
    #
    # We evaluate on ORIGINAL test data.
    #
    # We never apply SMOTE / Over Sampling /
    # Under Sampling to test data.

    results_df = evaluate_models(
        trained_models,
        X_test_processed,
        y_test
    )


    # Add balancing method so that we know
    # which method produced each result.

    results_df["balancing_method"] = method


    print("\n======================================")
    print("MODEL EVALUATION")
    print("======================================")


    print(
        results_df.to_string(
            index=False
        )
    )


    # Store results of current balancing method.

    all_results.append(
        results_df
    )



    # ========================================================
    # 13. STORE MODELS + MLFLOW
    # ========================================================

    for model_name, model in trained_models.items():


        # ----------------------------------------------------
        # Store trained model
        # ----------------------------------------------------

        all_trained_models[
            (method, model_name)
        ] = model



        # ====================================================
        # 14. START MLFLOW RUN
        # ====================================================

        # Example:
        #
        # smote - Random Forest
        # random_over - Random Forest
        # random_under - Random Forest

        start_run(
            f"{method} - {model_name}"
        )

        # ----------------------------------------------------
        # LOG MODEL PARAMETERS
        # ----------------------------------------------------

        params = model.get_params()


        log_parameters(
            params
        )



        # ----------------------------------------------------
        # GET METRICS FOR CURRENT MODEL
        # ----------------------------------------------------

        metrics_row = results_df[
            results_df["model"] == model_name
        ].iloc[0]



        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        metrics = {

            "accuracy":
                float(
                    metrics_row["accuracy"]
                ),

            "precision":
                float(
                    metrics_row["precision"]
                ),

            "recall":
                float(
                    metrics_row["recall"]
                ),

            "f1":
                float(
                    metrics_row["f1"]
                )
        }


        # ROC-AUC may not be available
        # for every model.

        if pd.notna(
            metrics_row["roc_auc"]
        ):

            metrics["roc_auc"] = float(
                metrics_row["roc_auc"]
            )


        print(
            f"\nLogging metrics for "
            f"{method} - {model_name}:"
        )


        print(metrics)


        log_metrics(
            metrics
        )



        # ----------------------------------------------------
        # LOG TAGS
        # ----------------------------------------------------

        tags = {

            "model_name":
                model_name,

            "imbalance_method":
                method,

            "experiment_type":
                "imbalance_comparison"
        }


        log_tags(
            tags
        )



        # ----------------------------------------------------
        # LOG MODEL
        # ----------------------------------------------------

        log_model(
            model,
            preprocessing_pipeline
        )



        # ----------------------------------------------------
        # END MLFLOW RUN
        # ----------------------------------------------------

        mlflow.end_run()



# ============================================================
# 15. COMBINE RESULTS FROM ALL BALANCING METHODS
# ============================================================

# At this point:
# 3 balancing methods x 4 ML models = 12 model combinations


final_results = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# 16. SORT ALL MODELS BY F1 SCORE
# ============================================================

# Highest F1 score will come first.

final_results = final_results.sort_values(
    by="f1",
    ascending=False
).reset_index(
    drop=True
)


print("\n======================================")
print("FINAL RESULTS - ALL 12 COMBINATIONS")
print("======================================")


print(
    final_results.to_string(
        index=False
    )
)


# ============================================================
# 17. SELECT BEST MODEL
# ============================================================

# Row 0 is now the model with the highest F1 score.

best_model_name = final_results.loc[
    0,
    "model"
]


best_method = final_results.loc[
    0,
    "balancing_method"
]



# Retrieve the correct trained model using both: balancing method + model name

best_model = all_trained_models[
    (
        best_method,
        best_model_name
    )
]


# ============================================================
# 18. PRINT BEST MODEL INFORMATION
# ============================================================

print("\n======================================")
print("BEST MODEL")
print("======================================")


print(
    "Balancing Method:",
    best_method
)


print(
    "Best Model:",
    best_model_name
)


print(
    "Accuracy:",
    round(
        final_results.loc[
            0,
            "accuracy"
        ],
        4
    )
)


print(
    "Precision:",
    round(
        final_results.loc[
            0,
            "precision"
        ],
        4
    )
)


print(
    "Recall:",
    round(
        final_results.loc[
            0,
            "recall"
        ],
        4
    )
)


print(
    "F1 Score:",
    round(
        final_results.loc[
            0,
            "f1"
        ],
        4
    )
)



# ROC-AUC

roc_auc_value = final_results.loc[
    0,
    "roc_auc"
]


if pd.notna(
    roc_auc_value
):

    print(
        "ROC-AUC:",
        round(
            roc_auc_value,
            4
        )
    )

else:

    print(
        "ROC-AUC: Not available"
    )



# ============================================================
# 19. SAVE PREPROCESSING PIPELINE
# ============================================================

# Save fitted preprocessing pipeline.
#
# app.py will use this preprocessing pipeline
# before making predictions.

joblib.dump(
    preprocessing_pipeline,
    "preprocessing_pipeline.pkl"
)


print(
    "\nPreprocessing pipeline saved successfully."
)



# ============================================================
# 20. SAVE BEST MODEL
# ============================================================

# This is the BEST model from
# all 12 combinations.

joblib.dump(
    best_model,
    "fraud_detection_model.pkl"
)


print(
    "Best model saved successfully:"
)


print(
    "fraud_detection_model.pkl"
)



# ============================================================
# 21. SAVE ALL MODEL EVALUATION RESULTS
# ============================================================

# This CSV will contain results from:3  balancing methods x 4 models = 12 rows

final_results.to_csv(
    "model_evaluation_results.csv",
    index=False
)


print(
    "\nModel evaluation results saved:"
)


print(
    "model_evaluation_results.csv"
)



# ============================================================
# 22. SAVE BEST MODEL INFORMATION
# ============================================================

best_model_information = pd.DataFrame([{

    "model":
        best_model_name,

    "balancing_method":
        best_method,

    "accuracy":
        final_results.loc[
            0,
            "accuracy"
        ],

    "precision":
        final_results.loc[
            0,
            "precision"
        ],

    "recall":
        final_results.loc[
            0,
            "recall"
        ],

    "f1":
        final_results.loc[
            0,
            "f1"
        ],

    "roc_auc":
        final_results.loc[
            0,
            "roc_auc"
        ]

}])


best_model_information.to_csv(
    "model_information.csv",
    index=False
)


print(
    "\nBest model information saved:"
)


print(
    "model_information.csv"
)



# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("======================================")


print(
    f"""

Best Balancing Method:
{best_method}

Best Model:
{best_model_name}

F1 Score:
{round(final_results.loc[0, "f1"], 4)}

Files Created:

1. preprocessing_pipeline.pkl

2. fraud_detection_model.pkl

3. model_evaluation_results.csv

4. model_information.csv

"""
)