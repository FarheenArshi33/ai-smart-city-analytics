import pandas as pd

# Load dataset
df = pd.read_csv("data/air_pollution.csv")

# First rows
print(df.head())

# Dataset info
print(df.info())

# Missing values
print(df.isnull().sum())

# Clean data
df = df.dropna()
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("Data cleaned successfully!")