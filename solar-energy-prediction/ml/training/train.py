import pandas as pd

# Load data
df = pd.read_csv("ml/data/raw/solar_training_5000.csv")

# ✅ Fix timestamp column
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Continue processing...