import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("../data/beijing_combined.csv")

# Drop missing
df = df.dropna()

# Sort by time
df = df.sort_values(by=["year","month","day","hour"])

# -------------------------
# CREATE LAG FEATURES
# -------------------------
z
# Past values of PM2.5
df['PM2.5_lag1'] = df['PM2.5'].shift(1)
df['PM2.5_lag2'] = df['PM2.5'].shift(2)
df['PM2.5_lag3'] = df['PM2.5'].shift(3)

# Drop rows with NA after shift
df = df.dropna()

# -------------------------
# FEATURES & TARGET
# -------------------------

X = df[['PM2.5_lag1','PM2.5_lag2','PM2.5_lag3','PM10','TEMP','WSPM']]
y = df['PM2.5']

# -------------------------
# TIME-BASED SPLIT (IMPORTANT)
# -------------------------

split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# -------------------------
# MODEL
# -------------------------

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -------------------------
# EVALUATION
# -------------------------

print("Time-Series Model Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))