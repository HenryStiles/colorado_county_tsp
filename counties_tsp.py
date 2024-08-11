import math
import pandas as pd
import time

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
start = counties[0]
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


TIMEOUT = 10  # seconds

# Brute force solution (invocation commented out).  Prints out the number of
# paths considered to show how hopeless this approach is.
import itertools
def TSP_brute(matrix, start):
    counties = matrix.counties.copy()
    counties.remove(start)
    min_cost = float("inf")
    min_path = None
    path_count = 0
    start_time = time.time()
    for path in itertools.permutations(counties):
        path_count += 1
        if path_count % 1000000 == 0:
            print("Paths considered:", path_count, "\r", end="")
        if time.time() - start_time > TIMEOUT:
            print("min cost:", min_cost)
            print("min path:", min_path)
            break
        cost = 0
        prev = start
        for county in path:
            cost += matrix.get_distance(prev, county)
            prev = county
        cost += matrix.get_distance(prev, start)
        if cost < min_cost:
            min_cost = cost
            min_path = path
    # add the start county to the min path list at the beginning and end.
    min_path = [start] + list(min_path) + [start]
    return min_path, min_cost

brute_tour, brute_cost = TSP_brute(matrix, start)
print("Bruce Force: ", brute_tour, brute_cost)


def TSP_NearestNeighbor(matrix, start):
    # make a copy of the counties list since we will modify it
    counties = matrix.counties.copy()
    # create a min path list initialized with the start county
    min_path = [start]
    # remove the start county from the counties list
    counties.remove(start)
    # loop until there are no more counties left
    total_cost = 0
    while counties:
        # get the last county in the min path list
        prev = min_path[-1]
        # find the nearest county to the last county
        min_county = None
        min_cost = float("inf")
        for county in counties:
            cost = matrix.get_distance(prev, county)
            if cost < min_cost:
                min_cost = cost
                min_county = county
        # add the nearest county to the min path list
        min_path.append(min_county)
        # remove the nearest county from the counties list
        counties.remove(min_county)
        total_cost += min_cost
    # add the cost to travel back to the start county
    total_cost += matrix.get_distance(min_path[-1], start)
    # add the start county to the min path list at the beginning and end.
    min_path = min_path + [start]
    return min_path, total_cost

nearest_tour, nearest_cost = TSP_NearestNeighbor(matrix, start)
print("Nearest Neighbor: ", nearest_tour, nearest_cost)


# branch and bound with minimum spanning tree. Nothing original here, much code used from https://www.geeksforgeeks.org/


# reduce cost matrix.
def reduce_matrix(matrix):
    n = len(matrix)
    lower_bound = 0

    # Row reduction
    for i in range(n):
        row_min = min(matrix[i])
        if row_min != float('inf') and row_min > 0:
            lower_bound += row_min
            for j in range(n):
                matrix[i][j] -= row_min

    # Column reduction
    for j in range(n):
        col_min = min(matrix[i][j] for i in range(n))
        if col_min != float('inf') and col_min > 0:
            lower_bound += col_min
            for i in range(n):
                matrix[i][j] -= col_min

    return lower_bound


# Branch and Bound, with reduced cost matrix
import time

def calculate_mst_cost(matrix, visited_indices):
    num_counties = len(matrix)
    unvisited = [i for i in range(num_counties) if i not in visited_indices]
    if not unvisited:
        return 0

    mst_cost = 0
    selected = [False] * num_counties
    key_values = [float('inf')] * num_counties
    key_values[unvisited[0]] = 0

    for _ in range(len(unvisited)):
        u = min((i for i in range(num_counties) if not selected[i] and i in unvisited), key=lambda i: key_values[i])
        selected[u] = True
        mst_cost += key_values[u]

        for v in range(num_counties):
            if matrix[u][v] != float('inf') and not selected[v] and matrix[u][v] < key_values[v]:
                key_values[v] = matrix[u][v]

    return mst_cost

def branch_and_bound(county_matrix):
    num_counties = len(county_matrix.counties)
    county_indices = list(range(num_counties))
    nn_tour, nn_cost = TSP_NearestNeighbor(county_matrix, county_matrix.counties[0])
    best_solution = {'tour': nn_tour, 'cost': nn_cost}
    start_time = time.time()
    # Statistics
    stats = {
        'total_nodes_visited': 0,
        'nodes_pruned': 0,
        'pruning_depths': {}
    }
    timeout_displayed = False
    def solve(matrix, current_index, visited_indices, current_cost, current_tour):
        nonlocal timeout_displayed
        if time.time() - start_time > TIMEOUT:
            if not timeout_displayed:
                timeout_displayed = True
                print("Timeout reached. Stopping algorithm.")
            return

        stats['total_nodes_visited'] += 1

        if len(visited_indices) == num_counties:
        # Complete the tour and update the best solution
            total_cost = current_cost + matrix[current_index][0]
            if total_cost < best_solution['cost']:
                best_solution['cost'] = total_cost
                best_solution['tour'] = [county_matrix.counties[index] for index in current_tour] + [county_matrix.counties[0]]
            return

        for next_index in county_indices:
            if next_index not in visited_indices:
                new_matrix = [row[:] for row in matrix]  # Copy matrix
                # Set the rows and columns for the current and next county to infinity
                for i in range(num_counties):
                    new_matrix[current_index][i] = float('inf')
                    new_matrix[i][next_index] = float('inf')
                # Prevent subtour
                new_matrix[next_index][0] = float('inf')
                # Calculate the reduction cost for the new matrix
                reduction_cost = reduce_matrix(new_matrix)
                mst_cost = calculate_mst_cost(new_matrix, visited_indices + [next_index])

                new_cost = current_cost + matrix[current_index][next_index] + reduction_cost + mst_cost
                if new_cost < best_solution['cost']:
                    solve(new_matrix, next_index, visited_indices + [next_index], new_cost, current_tour + [next_index])
                else:
                    stats['nodes_pruned'] += 1
                    stats['pruning_depths'][len(visited_indices)] = stats['pruning_depths'].get(len(visited_indices), 0) + 1
    # Initial matrix reduction
    initial_matrix = [[county_matrix.get_distance(county_matrix.counties[i], county_matrix.counties[j]) for j in county_indices] for i in county_indices]
    initial_reduction_cost = reduce_matrix(initial_matrix)
    solve(initial_matrix, 0, [0], initial_reduction_cost, [0])
    print("Total nodes visited:", stats['total_nodes_visited'], "Nodes pruned:", stats['nodes_pruned'], "Pruning depths:", stats['pruning_depths'])
    return best_solution['tour'], best_solution['cost']


branch_tour, branch_cost = branch_and_bound(matrix)
print("Branch and Bound: ", branch_tour, branch_cost)
