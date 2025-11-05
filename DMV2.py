#go to openweathermap login and get api key 
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_KEY='a49142b21e4ce14f607740acec645169'
city=input("Enter the city:")

url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
weather_data = response.json()

if response.status_code==200:
    main = weather_data['main']
    wind = weather_data.get('wind',{})
    weather_info = { 
            'City': city,
            'Temperature': main.get('temp'),
            'Feels like (C)': main.get('feels_like'),
            'Humidity (%)': main.get('humidity'),
            'Pressure (hPa)': main.get('pressure'),
            'Wind speed (m/s)': main.get('speed')
    }
else:
    print(f"Error fetching data from {city}: {weather_data.get('message')}")
    exit()

weather_df= pd.DataFrame([weather_info])
print('\nWeather Data:\n', weather_df)

print("\nDescriptive Statistics:\n", weather_df.describe())

if weather_df.isnull().any().any():
    print("Some weather attributes were missing and have been filled with 0")


weather_df= weather_df.fillna(0)
plt.figure(figsize=(8,4))
plt.bar(weather_df.columns[1:], weather_df.iloc[0,1:], color='green')
plt.title("Weather conditions in city")
plt.xlabel("Attributes")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.legend()
plt.show()
