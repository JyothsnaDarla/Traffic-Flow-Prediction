from flask import Flask, render_template, request
import sqlite3
import numpy as np
import folium
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import pytz
import string

# Initialize Flask app
app = Flask(__name__)
nltk.download('stopwords')

# Load and preprocess dataset
df = pd.read_csv("merged_dataset.csv")  # Ensure your dataset path is correct
df = df.drop_duplicates()

# Geolocation function with error handling
def preprocess_location(location_str):
    geolocator = Nominatim(user_agent="traffic_prediction_app", timeout=10)
    try:
        location = geolocator.geocode(location_str)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            raise ValueError("Location not found.")
    except Exception as e:
        print(f"Geocoding error: {e}")
        raise ValueError("Geocoding service is currently unavailable. Please try again later.")

# Updated text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to text data in the dataframe
df['text'] = df['text'].apply(preprocess_text)

# Split data
X = df[['latitude', 'longitude', 'text']]
y = df['trafficflow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines
geo_pipeline = make_pipeline(
    ColumnTransformer([('geo', StandardScaler(), ['latitude', 'longitude'])]),
    RandomForestClassifier(random_state=42)
)

# Text pipeline with hyperparameter tuning (using GridSearchCV)
text_pipeline = make_pipeline(
    TfidfVectorizer(),
    RandomForestClassifier(random_state=42)
)

# Hyperparameter tuning for text pipeline
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [10, 20]
}

text_grid_search = GridSearchCV(text_pipeline, param_grid, cv=5)
text_grid_search.fit(X_train['text'], y_train)

# Train the geographical model
geo_pipeline.fit(X_train[['latitude', 'longitude']], y_train)

# Initialize SQLite database
DATABASE = 'traffic_flow.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT,
            street_name TEXT,
            description TEXT,
            traffic_flow_prediction REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/entry', methods=['GET', 'POST'])
def entry():
    error_message = None
    if request.method == 'POST':
        location = request.form['location'].strip().lower()  # Convert to lowercase
        street_name = request.form.get('street_name', '').strip().lower()  # Convert to lowercase
        description = request.form['description'].strip().lower()

        try:
            lat, lon, address = preprocess_location(f"{location}, {street_name}")
            input_data = pd.DataFrame([[lat, lon]], columns=['latitude', 'longitude'])
            geo_prediction = geo_pipeline.predict(input_data)[0]
            text_processed = preprocess_text(description)
            text_prediction = text_grid_search.predict([text_processed])[0]

            similar_cases = df[(df['latitude'] == lat) & (df['longitude'] == lon) & (df['text'] == text_processed)]
            if not similar_cases.empty:
                historical_flow = similar_cases['trafficflow'].mean()
                final_prediction = (geo_prediction + text_prediction + historical_flow) / 3
            else:
                final_prediction = (geo_prediction + text_prediction) / 2

            traffic_status = "Smooth Traffic" if final_prediction < 0.4 else "Moderate Traffic" if final_prediction < 0.7 else "Heavy Traffic"

            timestamp_ist = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO traffic_data (location, street_name, description, traffic_flow_prediction, timestamp) VALUES (?, ?, ?, ?, ?)",
                           (location, street_name, description, final_prediction, timestamp_ist))
            conn.commit()
            conn.close()

            # Construct the full address without lat/lon
            full_address = f"{location}, {street_name}"
            map_file_path = create_map(lat, lon, full_address)

            return render_template('result.html', 
                                   location=full_address, 
                                   description=description, 
                                   geo_prediction=geo_prediction, 
                                   text_prediction=text_prediction, 
                                   final_prediction=final_prediction, 
                                   traffic_status=traffic_status, 
                                   map_file_path=map_file_path)

        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            error_message = "An unexpected error occurred. Please try again later."

    return render_template('entry.html', error_message=error_message)

@app.route('/show', methods=['GET', 'POST'])
def show():
    data = None
    error_message = None

    if request.method == 'POST':
        location = request.form['location'].strip().lower()  # Convert to lowercase
        street_name = request.form.get('street_name', '').strip().lower()
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        try:
            if street_name:
                cursor.execute(""" 
                    SELECT * FROM traffic_data 
                    WHERE location = ? AND street_name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 5 
                """, (location, street_name))
            else:
                cursor.execute(""" 
                    SELECT * FROM traffic_data 
                    WHERE location = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 5 
                """, (location,))

            data = cursor.fetchall()

            if data:
                for i in range(len(data)):
                    prediction_value = data[i][4]  # Assuming 'traffic_flow_prediction' is in the 5th column
                    prediction = 'Smooth' if prediction_value < 0.5 else 'Moderate' if prediction_value < 1.5 else 'Heavy'
                    data[i] = list(data[i])
                    data[i][4] = prediction
            else:
                error_message = "No data found for the specified location."

        except Exception as e:
            error_message = "An unexpected error occurred while retrieving data."
            print(f"Error: {e}")
        finally:
            conn.close()

    return render_template('show.html', data=data, error_message=error_message)

def create_map(lat, lon, address):
    map_ = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker(location=[lat, lon], tooltip="Predicted Traffic Flow Area", popup=address).add_to(map_)
    map_file_path = "static/map.html"
    map_.save(map_file_path)
    return map_file_path

if __name__ == '__main__':
    app.run(debug=True)
