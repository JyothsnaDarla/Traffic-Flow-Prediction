 ***Traffic Flow Prediction***
 ---
 
This project aims to provide a real-time traffic prediction system that uses geolocation (latitude and longitude) and user-described traffic situations to forecast traffic flow. By integrating machine learning and natural language processing (NLP), the app helps users plan safer and more efficient routes across different locations.

## Directions to use this repo:

- Clone the repository by running:  
  `git clone https://github.com/JyothsnaDarla/traffic-prediction.git`
- Install all dependencies:  
  `pip install -r requirements.txt`
- Run the application:  
  `python app.py`
- Visit the app in your browser:  
  `http://127.0.0.1:5000/`

***How It Works***
---

The user enters a city name, optionally a street name, and a traffic-related description (e.g., "construction", "evening rush", "accident near market").
The system extracts the latitude and longitude using the Geopy library.
Two machine learning models are used:
A text-based model using TF-IDF and Random Forest
A location-based model using latitude and longitude
The final prediction combines both model results (and historical data if available) to classify traffic as:
Smooth Traffic,Moderate Traffic,Heavy Traffic
Results are displayed on an interactive map, and also stored in an SQLite database.

 ***Key Features***
 ---
Real-time traffic prediction using both location and description
NLP preprocessing using NLTK
Geolocation with Nominatim (Geopy)
Map visualization using Folium
Historical data tracking via SQLite
Easy-to-use web interface built with Flask

**Home page:**
Introduction to the application
![Image](https://github.com/user-attachments/assets/1015e9e0-5fd5-4007-ac21-b059c667f83b)
**Entry page:**
Form to input traffic details and get prediction
![Image](https://github.com/user-attachments/assets/88f5b6e5-45e0-4654-9ed8-77d60f9d0ea9)
**result page:**
Shows traffic prediction and map
![Image](https://github.com/user-attachments/assets/a9025736-fe2c-4d5c-913e-1c5feb98af9d)
**show page:**
Displays recent predictions for selected location
![Image](https://github.com/user-attachments/assets/ccb5c4ac-23cf-4d50-93fc-c738915c9b08)

**Objective:**
---
This project aims to reduce traffic-related stress by helping users avoid congested routes. It supports smarter commuting decisions and promotes better traffic flow management using a data-driven approach.
