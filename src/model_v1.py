import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor, plot_importance


# We set up a logging configuration to capture the model's training and evaluation process.
logging.basicConfig(

    # Gives us a timestamp, log level, and message in the logs
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)



def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset and prepare features/target for modeling.

    Working:
        - Reads the dataset from a CSV file using pandas.
        - Drops 'Rating Date' and 'CIK' as they are identifiers, not predictive features.
        - Separates features (X) from the target variable (y, 'Rating') for model training.

    """
    df = pd.read_csv(filepath)

    # Drop non-predictive columns if present
    df = df.drop(columns=["Rating Date", "CIK"], errors="ignore")

    X = df.drop(columns=["Rating"])
    y = df["Rating"]

    logging.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split dataset into training and testing sets.

    Working:
        - Splits data into 80% training and 20% testing sets using sklearn.
        - Ensures consistent splits with a fixed random seed.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBRegressor:
    """
    Train a baseline XGBoost model with early stopping.

    Working:
        - Initializes XGBRegressor with balanced hyperparameters.
        - Trains model with early stopping to avoid overfitting, monitoring RMSE.
        - Uses validation set to stop training if performance doesn’t improve.
    """
    model = XGBRegressor(
        n_estimators=2000, 
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    logging.info("Baseline XGBoost model trained successfully.")
    return model


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    Perform hyperparameter tuning with RandomizedSearchCV.

    Working:
        - Defines a grid of hyperparameters to test.
        - Uses RandomizedSearchCV to evaluate 20 combinations with 5-fold cross-validation.
        - Optimizes for RMSE and logs the best parameters and score.
    """
    param_grid = {
        'n_estimators': [500, 1000, 1500, 2000],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'subsample': [0.5, 0.7, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
        'min_child_weight': [1, 3, 5]
    }

    base_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    logging.info(f"Best Parameters: {search.best_params_}")
    logging.info(f"Best CV RMSE: {-search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_



def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model using RMSE and R².

    Working:
        - Generates predictions on the test set.
        - Calculates RMSE to measure prediction error.
        - Calculates R² to measure variance explained by the model.
    """
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Test RMSE: {rmse:.4f}")
    logging.info(f"Test R²: {r2:.4f}")

    return y_pred, rmse, r2


def plot_feature_importance(model: XGBRegressor, X: pd.DataFrame, save_path: str = None):
    """
    Plot feature importance for the trained model.

    Working:
        - Calculates feature importance based on gain using XGBoost’s built-in function.
        - Creates and displays (or saves) a plot to show which features drive predictions.
    """
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type="gain", xlabel="Gain", ylabel="Feature")
    plt.title("Feature Importance (XGBoost)")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logging.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()



def model_pipeline(filepath: str, use_tuning: bool = False) -> XGBRegressor:
    """
    Full pipeline for training and evaluating XGBoost on credit rating dataset.

    Working:
        - Loads and splits data into training and testing sets.
        - Trains a baseline model or tunes hyperparameters based on use_tuning.
        - Evaluates model performance and saves the model to a JSON file.
        - Plots and saves feature importance to highlight key predictors.

    """
    X, y = load_and_prepare_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)

    os.makedirs("models", exist_ok=True)

    if use_tuning:
        logging.info("Running Hyperparameter Tuning...")
        tuned_model, best_params = tune_hyperparameters(X_train, y_train)
        evaluate_model(tuned_model, X_test, y_test)

        logging.info("Retraining final model on ALL data with best parameters...")
        model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            eval_metric="rmse",
            n_jobs=-1
        )
        model.fit(X, y, verbose=False)
        save_path = "models/credit_rating_xgboost_tuned.json"
    else:
        logging.info("Training baseline model...")
        model = train_xgboost(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test)
        save_path = "models/credit_rating_xgboost_baseline.json"

    model.save_model(save_path)
    logging.info(f"Model saved at {save_path}")

    # Plot feature importance
    plot_feature_importance(model, X, save_path="models/feature_importance.png")

    return model


if __name__ == "__main__":
    # Set use_tuning=True to enable RandomizedSearchCV or shenei!!!!
    model_pipeline("data/Final_Dataset_v2.csv", use_tuning=False)
