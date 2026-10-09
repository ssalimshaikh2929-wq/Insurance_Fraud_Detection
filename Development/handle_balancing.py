from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def handle_imbalance(
    X_train,
    y_train,
    method = "smote"
):

    if method == "smote":

        sampler = SMOTE()

    elif method == "random_over":

        sampler = RandomOverSampler()

    elif method == "random_under":

        sampler = RandomUnderSampler()
        

    else:

        raise ValueError(
            "Invalid imbalance handling method. "
            "Use: smote, random_over, random_under"
        )

    X_train_resampled, y_train_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    return X_train_resampled, y_train_resampled