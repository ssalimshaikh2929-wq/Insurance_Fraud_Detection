import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_models(trained_models, X_test, y_test):

    # Store evaluation results of all models
    results = []

    # Loop through each trained model
    for name, model in trained_models.items():

        # Predict Fraud (1) or Not Fraud (0)
        y_pred = model.predict(X_test)


        # ------------------------------------------
        # Accuracy
        # Overall percentage of correct predictions.
        #
        # (TP + TN) / Total Predictions
        #
        # Be careful:
        # Accuracy can be misleading for imbalanced
        # fraud datasets.
        # ------------------------------------------
        accuracy = accuracy_score(y_test, y_pred)


        # ------------------------------------------
        # Precision
        # Out of all transactions predicted as Fraud,
        # how many were actually Fraud?
        #
        # TP / (TP + FP)
        #
        # High precision = fewer false fraud alerts.
        # ------------------------------------------
        precision = precision_score(y_test, y_pred, zero_division=0)  


        # ------------------------------------------
        # Recall
        # Out of all actual Fraud cases,
        # how many did our model successfully detect?
        #
        # TP / (TP + FN)
        #
        # Very important in fraud detection because
        # low recall means we're missing fraud cases.
        # ------------------------------------------
        recall = recall_score(y_test, y_pred, zero_division=0)


        # ------------------------------------------
        # F1 Score
        # Balance between Precision and Recall.
        #
        # Useful for imbalanced classification.
        # ------------------------------------------
        f1 = f1_score(y_test, y_pred, zero_division=0)


        # ------------------------------------------
        # ROC-AUC
        # Measures how well the model separates
        # Fraud (1) from Not Fraud (0).
        #
        # Uses probabilities instead of only
        # final 0/1 predictions.
        # ------------------------------------------

        if hasattr(model, "predict_proba"):

            y_probability = model.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                y_probability
            )

        else:
            roc_auc = None


        # Store results
        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })


    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    return results_df