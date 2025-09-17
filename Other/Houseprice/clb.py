# Install if needed (usually already present in Colab)
# !pip install pandas numpy scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# 🔹 If you uploaded train.csv directly to Colab (left sidebar -> Files)
# Replace path if you placed it in Google Drive
df = pd.read_csv("train.csv")  # Kaggle House Prices dataset

df.head()

features = ["OverallQual", "GrLivArea", "GarageCars",
            "TotalBsmtSF", "FullBath", "YearBuilt"]
target = "SalePrice"

data = df[features + [target]].dropna()

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print("R² score on test set:", round(model.score(X_test, y_test), 3))
