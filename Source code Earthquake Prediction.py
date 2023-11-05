# Import Necessary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor  # Import MLPRegressor
import cartopy.crs as ccrs  # Import cartopy for map creation

# Read Dataset
data = pd.read_csv(r"./database.csv")
print(data.head())

# Data Preprocessing
data['Date_Time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
data = data.dropna(subset=['Date_Time'])
print(data)

# Remove rows with missing 'Date_Time'
data = data[['Date_Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
print(data)

# Calculate Unix timestamps
timestamp = data['Date_Time'].apply(lambda x: x.timestamp())
data['Timestamp'] = timestamp
final_data = data[['Timestamp', 'Latitude', 'Longitude', 'Magnitude', 'Depth']]
print(final_data)

# Splitting the Data
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test )

# Train Random Forest Regressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
print("Random Forest Regressor Score:", score)

# Hyperparameter tuning
parameters = {'n_estimators': [10, 20, 50, 100, 200, 500]}
grid_obj = GridSearchCV(reg, parameters)
grid_fit = grid_obj.fit(X_train, y_train)
best_fit = grid_fit.best_estimator_
score = best_fit.score(X_test, y_test)
print("Best Random Forest Regressor Score:", score)

# Neural Network model
model = MLPRegressor(hidden_layer_sizes=(16, 16), activation='relu', solver='adam', max_iter=1000, random_state=1)
model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)
print("Neural Network Model Score:", test_score)

# Visualize Earthquake Locations
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.stock_img()

longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()

ax.plot(longitudes, latitudes, 'ro', markersize=2)
plt.title("Earthquake Locations")
plt.show()

#Thus the End of Program!...
