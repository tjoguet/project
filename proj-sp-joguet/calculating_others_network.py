import numpy as np
from calculating_euclidean import calculate_distances


class Others_Network:
    def __init__(self):
        # This function will define the different hyperparameters
       # self.distances = calculate_distances('data/SiouxFalls_node-2.tntp')
        self.coeff_euclidean_distance_to_taxi_time = 13/388329.757
        self.coeff_euclidean_distance_to_taxi_price = 12.84*2.73/388329.757
        self.coeff_euclidean_distance_to_walking_time = 12.84/4.5/3/388329.757*60
        with open('data/euclidean.txt', 'r') as file:
            # Read the contents of the file
            self.euclidean = file.readlines()
    
    def calculate_time_and_price(self, node1, node2):
        # This function will calculate the time it takes and the price to travel from node1 to node2 using a taxi
        euclidean_distance = self.get_euclidean_distance(node1, node2)
   #     print("Euclidean distance:", euclidean_distance)
        taxi_time = euclidean_distance * self.coeff_euclidean_distance_to_taxi_time
        taxi_price = euclidean_distance * self.coeff_euclidean_distance_to_taxi_price
        walking_time = euclidean_distance * self.coeff_euclidean_distance_to_walking_time

        return taxi_time, taxi_price, walking_time

    def get_euclidean_distance(self, node1, node2):
        # This function will find the euclidean distance between two nodes
       # print("Node1:", node1)
       # print("Node2:", node2)
     
        value = float(self.euclidean[node1].split(',')[node2])   
        
        
        #print("Euclidean distance:", value)  
        return value


