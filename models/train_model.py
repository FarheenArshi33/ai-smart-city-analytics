import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")

# Features
X = df[[
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value"
]]

# Target
y = df["AQI Value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", error)

# Save model
joblib.dump(model, "models/aqi_model.pkl")

print("Model saved successfully!")