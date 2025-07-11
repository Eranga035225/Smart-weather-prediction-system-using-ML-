import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load datasets
location_df = pd.read_csv("data/locationData.csv")
weather_df = pd.read_csv("data/weatherData.csv")

# Merge on location_id
merged_df = pd.merge(weather_df, location_df, on="location_id", how="left")

# Convert 'date' column to datetime
merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

# Drop unnecessary columns
columns_to_drop = ["sunrise", "sunset", "timezone", "timezone_abbreviation"]
merged_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Extract date features
merged_df['month'] = merged_df['date'].dt.month
merged_df['day'] = merged_df['date'].dt.day
merged_df['dayofweek'] = merged_df['date'].dt.dayofweek

# Label encode city names
le = LabelEncoder()
merged_df['city_name'] = le.fit_transform(merged_df['city_name'])

# Fill missing values
merged_df.fillna(method='ffill', inplace=True)

# Optional: Save the cleaned and merged dataset
merged_df.to_csv("data/merged_cleaned_weather_data.csv", index=False)

print("Preprocessing complete. File saved as 'data/merged_cleaned_weather_data.csv'")
