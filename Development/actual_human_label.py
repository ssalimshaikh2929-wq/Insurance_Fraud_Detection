from model_monitoring import add_actual_label


# ---------------------------------------------
# Path of prediction log file
# ---------------------------------------------

log_file = "prediction_logs.csv"


# ---------------------------------------------
# Actual labels confirmed by human/investigator
#
# 1 = Fraud
# 0 = Not Fraud
#
# IMPORTANT:
# Number of labels must match number of rows
# inside prediction_logs.csv
# ---------------------------------------------

actual_labels = [
    1,
    0,
    1,
    0,
    0
]


# ---------------------------------------------
# Add actual labels to prediction log
# ---------------------------------------------

add_actual_label(
    log_file,
    actual_labels
)


print("Actual labels added successfully.")