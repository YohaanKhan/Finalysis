import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. Reload the dataset
df = pd.read_csv("data/AAPL_structured.csv")
X = df.drop(columns=["Rating Date"], errors="ignore")


# 2. Load the trained model
loaded_model = XGBRegressor()
loaded_model.load_model("models/credit_rating_xgboost_final.json")   

# 3. Make predictions
y_pred = loaded_model.predict(X)

# 4. Evaluate
print(f"The model predicted ratings: {y_pred[:5]}")  # Display first prediction