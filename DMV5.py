import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
data = pd.read_csv('C:/Users/User 58/Downloads/AirQuality.csv')

# Explore the dataset
print(data.head())
print(data.info())

# Convert Date column
# Convert Date column safely
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Handle invalid dates
data['Date'].fillna(method='ffill', inplace=True)


# 1. Line plot for AQI trend
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['AQI'], label='AQI', color='blue')
plt.xlabel('Date')
plt.ylabel('Air Quality Index (AQI)')
plt.title('AQI Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()

# 2. Pollutant Trends
for col, color in [('PM2.5','green'), ('PM10','orange'), ('CO','red')]:
    plt.figure(figsize=(10,6))
    plt.plot(data['Date'], data[col], color=color, label=col)
    plt.title(f'{col} Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel(f'{col} Levels')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# 3. Bar chart - Monthly average AQI
data['Month'] = data['Date'].dt.to_period('M')
avg_aqi_per_month = data.groupby('Month')['AQI'].mean().reset_index()

plt.figure(figsize=(12, 4))
plt.bar(avg_aqi_per_month['Month'].astype(str), avg_aqi_per_month['AQI'], color='purple')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Average AQI')
plt.title('Average AQI per Month')
plt.grid(True)
plt.show()

# 4. Boxplot for pollutant distribution
plt.figure(figsize=(8,4))
sns.boxplot(data=data[['PM2.5', 'PM10', 'CO']])
plt.title('Distribution of Pollutant Levels')
plt.ylabel('Concentration')
plt.grid(True)
plt.show()

# 5. Scatter plot: AQI vs Pollutants
plt.figure(figsize=(10, 5))
plt.scatter(data['PM2.5'], data['AQI'], label='PM2.5', alpha=0.6, color='red')
plt.scatter(data['PM10'], data['AQI'], label='PM10', alpha=0.6, color='green')
plt.scatter(data['CO'], data['AQI'], label='CO', alpha=0.6, color='blue')
plt.xlabel('Pollutant Level')
plt.ylabel('AQI')
plt.title('Relationship between AQI and Pollutant Levels')
plt.legend()
plt.grid(True)
plt.show()
