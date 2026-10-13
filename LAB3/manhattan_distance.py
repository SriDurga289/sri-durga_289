import numpy as np
from scipy.spatial import distance

pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

manhattan_dist = distance.cityblock(pointA, pointB)
print("Manhattan Distance:", manhattan_dist)

similarity_manhattan = 1 / (1 + manhattan_dist)
print("Manhattan Similarity:", similarity_manhattan)