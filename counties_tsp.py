import math
import pandas as pd

# data file from the State of Colorado, county seats and coordinates.
input_file = 'Colorado_County_Seats_20240114.csv'

# Measure distance between two points on the Earth.
def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    # convert to miles
    distance = distance * 0.621371
    return distance


df = pd.read_csv(input_file)

location_data = df.iloc[:,5]

# location is in the format County Name, newline, "latitude, longitude"
# make a dictionary of county names and coordinates
county_coordinates = {}
for location in location_data:
    county_name, coordinates = location.split('\n')
    # remove parentheses from the coordinates
    coordinates = coordinates.replace('(', '').replace(')', '')
    latitude, longitude = coordinates.split(',')
    # convert to floats
    latitude = float(latitude)
    longitude = float(longitude)
    county_coordinates[county_name] = (latitude, longitude)

print(county_coordinates)

