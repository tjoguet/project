import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 
    # c= acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1))
    c = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))    
    r = 6371  # Radius of Earth in kilometers
    distance = r * c

    return distance

def calculate_distances(filename):
    with open(filename, 'r') as file:
        table_contents = file.read()
    lines = table_contents.strip().split('\n')

    coordinates = [0] * (len(lines)-1)
    for i in range(1, len(lines)):
       coordinates[i-1] = (float(lines[i].split('\t')[1]), float(lines[i].split('\t')[2]))
    print(coordinates[0])
    distances = []
    # Calculate distances between each pair of coordinates
    for i in range(len(coordinates)):
        row_distances = []  # Initialize a row to store distances for each coordinate pair
        for j in range(len(coordinates)):
            if i == j:
                row_distances.append(0.0)  # Distance from a coordinate to itself is 0
            elif j < i:  # Use the already calculated distance (since it's symmetric)
                row_distances.append(distances[j][i])
            else:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                dist_km = haversine(lat1, lon1, lat2, lon2)
                row_distances.append(dist_km)  # Append the distance to the row
        distances.append(row_distances)  # Append the row to the distances list
    print(distances[0])
    return distances

if __name__ == "__main__":
    distances = calculate_distances('data/SiouxFalls_node-2.tntp')
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            print(f"Distance between node {i+1} and node {j+1}: {distances[i][j]:.2f} km")
