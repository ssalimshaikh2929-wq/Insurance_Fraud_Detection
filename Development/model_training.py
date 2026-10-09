from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train):
    
    # Train multiple machine learning models on the training data.

    # Parameters:
    # - X_train: Features for training.
    # - y_train: Target variable for training.

    # Returns:
    # - A dictionary containing trained models.
    # """
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    # Train each model
    
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"{name} trained successfully.")
    
    return trained_models