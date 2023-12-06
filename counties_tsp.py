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


def test_haversine():
    # get coordinates for boulder
    blat, blong = county_coordinates['Boulder, CO']
    print(blat,blong)
    for county in county_coordinates:
        # get the destination city coords.
        dlat, dlong = county_coordinates[county]
        # calculate the distance
        dist = haversine(blat, blong, dlat, dlong)
        print("distance from Boulder to " + county + " = ", dist)


# DS to hold the counties and distances between them.
class CountyDistanceMatrix:
    def __init__(self, counties):
        self.counties = counties
        self.index = {county: index for index, county in enumerate(counties)}
        self.matrix = [[float('inf') for _ in counties] for _ in counties]

    def add_distance(self, county1, county2, distance):
        i, j = self.index[county1], self.index[county2]
        self.matrix[i][j] = distance
        self.matrix[j][i] = distance  # Assuming distance is symmetric

    def get_distance(self, county1, county2):
        try:
            i, j = self.index[county1], self.index[county2]
        except KeyError:
            print(f"KeyError: One of the counties {county1} or {county2} does not exist in the index.")
            return None
        return self.matrix[i][j] 

    def __str__(self):
        matrix_str = "\t" + "\t".join(self.counties) + "\n"
        for i, row in enumerate(self.matrix):
            matrix_str += self.counties[i] + "\t" + "\t".join(map(str, row)) + "\n"
        return matrix_str
        
counties = list(county_coordinates.keys())
# move Boulder to the front of the list (start city)
counties.remove('Boulder, CO')
counties = ['Boulder, CO'] + counties
matrix = CountyDistanceMatrix(counties)

# Add distances between all counties
for i in range(len(counties)):
    for j in range(i + 1, len(counties)):
        county1 = counties[i]
        county2 = counties[j]
        lat1, lon1 = county_coordinates[county1]
        lat2, lon2 = county_coordinates[county2]
        distance = haversine(lat1, lon1, lat2, lon2)
        matrix.add_distance(county1, county2, distance)



print(matrix)
