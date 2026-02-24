import pandas as pd
import numpy as np

# 1. Setup Timeframe (5000 intervals = ~52 days)
time_index = pd.date_range(start="2026-02-16", periods=5000, freq="15min")
df = pd.DataFrame(index=time_index)

# 2. Plant Technical Constants [cite: 18, 21]
AC_CAPACITY = 1.0  # 1000 KW Limit
DC_AC_RATIO = 1.2  # 1200 KWp DC Capacity
df['Installed_AC_MW'] = AC_CAPACITY
df['DC_Capacity_MWp'] = AC_CAPACITY * DC_AC_RATIO

# 3. Time & Astronomical Features 
df['Hour_of_Day'] = df.index.hour
df['Day_of_Year'] = df.index.dayofyear
# Solar Elevation: 90 deg at noon, negative at night
df['Solar_Elev_Angle'] = 90 * np.sin(np.pi * (df['Hour_of_Day'] - 6) / 12)
df.loc[df['Solar_Elev_Angle'] < 0, 'Solar_Elev_Angle'] = 0 

# 4. Weather Data [cite: 20]
# Randomizing cloud cover to create "Ramp Events"
df['Cloud_Cover_Pct'] = np.random.choice([0, 10, 30, 85], size=5000, p=[0.5, 0.2, 0.2, 0.1])
df['Amb_Temp_C'] = 22 + 8 * np.sin(np.pi * (df['Hour_of_Day'] - 8) / 12) + np.random.normal(0, 1, 5000)
df['GHI'] = np.where(df['Solar_Elev_Angle'] > 0, 
                     1000 * np.sin(np.pi * (df['Hour_of_Day'] - 6) / 12) * (1 - df['Cloud_Cover_Pct']/100), 0)
df['DNI'] = df['GHI'] * 0.75
df['DHI'] = df['GHI'] * 0.25
df['Wind_Speed'] = np.random.uniform(1, 6, 5000)
df['Humidity'] = np.random.uniform(20, 80, 5000)
df['Pressure'] = 1013 + np.random.normal(0, 2, 5000)
df['Visibility'] = 10 - (df['Cloud_Cover_Pct'] / 10)

# 5. Generation Logic (Inc. Temperature Losses & Midday Clipping) [cite: 17, 18]
# Efficiency drops 0.4% per degree above 25°C
efficiency = 1.0 - np.maximum(0, (df['Amb_Temp_C'] - 25) * 0.004)
dc_gen = (df['GHI'] / 1000) * df['DC_Capacity_MWp'] * efficiency
df['Actual_Power_MW'] = np.minimum(AC_CAPACITY, dc_gen) # Clipping Plateau

# 6. Derived Engineering Features 
df['Lag_1'] = df['Actual_Power_MW'].shift(1).fillna(0)
df['Lag_24'] = df['Actual_Power_MW'].shift(96).fillna(0) # 96 intervals = 24h
df['Lag_48'] = df['Actual_Power_MW'].shift(192).fillna(0)
df['Rolling_Mean_24h'] = df['Actual_Power_MW'].rolling(window=96).mean().fillna(0)
df['Ramp_Rate'] = df['Actual_Power_MW'].diff().fillna(0)
df['Capacity_Factor'] = df['Actual_Power_MW'] / AC_CAPACITY

# 7. Operational Metadata 
df['Inverter_Clipping_Flag'] = (dc_gen > AC_CAPACITY).astype(int)
df['Grid_Availability'] = 1
df['Maintenance_Log'] = 0
df['Tracking_Type'] = 1 # Single-axis

# Save Output
df.to_csv("solar_training_5000.csv")
print(f"Generated {len(df)} rows of data with 25 features.")