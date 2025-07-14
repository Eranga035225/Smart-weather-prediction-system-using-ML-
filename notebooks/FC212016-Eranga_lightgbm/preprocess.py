import pandas as pd

def preprocess_weather_and_location():
    # Load CSV files
    weather = pd.read_csv('data/weather.csv')
    locations = pd.read_csv('data/locations.csv')

    # Merge on location_id to get location info in weather data
    df = pd.merge(weather, locations, on='location_id', how='left')

    # Convert date column to datetime (handle mixed formats, dayfirst)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')

    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Convert sunrise and sunset to hours (some may be missing, so use errors='coerce')
    df['sunrise_hour'] = pd.to_datetime(df['sunrise'], format='%H:%M', errors='coerce').dt.hour
    df['sunset_hour'] = pd.to_datetime(df['sunset'], format='%H:%M', errors='coerce').dt.hour
    df['daylight_hours'] = df['sunset_hour'] - df['sunrise_hour']

    # Drop original date, sunrise, sunset columns (optional)
    df.drop(columns=['date', 'sunrise', 'sunset'], inplace=True)

    # Drop columns you don’t need or are redundant
    df.drop(columns=['utc_offset_seconds', 'timezone', 'timezone_abbreviation'], inplace=True)

    # Handle missing values: drop or fill (here drop)
    df = df.dropna()

    # One-hot encode city_name
    df = pd.get_dummies(df, columns=['city_name'], drop_first=True)

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Save cleaned data
    df.to_csv('data/cleaned_weather.csv', index=False)
    print("✅ Preprocessing complete! Saved to data/cleaned_weather.csv")

if __name__ == "__main__":
    preprocess_weather_and_location()
