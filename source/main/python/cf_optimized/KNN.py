# Imports
import math
from operator import itemgetter

class KNN(object):
    """k-nearest neighbor class."""

    def __init__(self, trainingSet, testInstance, k):
        """Returns a new KNN object.
        :param trainingSet: set representing the training data.
        :param testinstance: vector of float numbers representing features of response variable.
        :param k:  number of nearest neighbors used for prediction test instance."""
        self.trainingSet = trainingSet
        self.testInstance = testInstance
        self.k = k
        self.getNearestNeighbors(trainingSet, testInstance, k)


    def euclidianDistance(self, instance1, instance2, dimension):
        """ Returns the euclidian distance between two data instances.
        :param instance1: training data instance
        :param instance2: unseen sample instance
        :param dimension: feature dimension"""
        distance = 0
        for x in range(dimension): # length is dimension of feature
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


    def getNearestNeighbors(self, trainingSet, testInstance, k):
        """ Returns the k nearest neighbors of an unseen sample instance.
        @:param trainingSet: training set
        @:param testInstance: unseen sample instance
        @:param k: number of nearest neighbors to return"""
        distances = []
        dimension = len(testInstance) # dimension of features, last column is response variable
        for x in range(len(trainingSet)):
            dist = self.euclidianDistance(testInstance, trainingSet[x], dimension)
            distances.append((trainingSet[x][dimension], dist))
        distances.sort(key=itemgetter(1)) # sorts distances in O(nlogn). (Replace with randomized quickselect)
        neighbors = distances[:k]
        self.getPrediction(neighbors)
        return neighbors


    def getPrediction(self, neighbors):
        """" Returns prediction of response variable (inverse distance weighting).
        @:param neighbors: list of (response, distance)-tuples."""
        responses = map(itemgetter(0), neighbors)
        distances = map(itemgetter(1), neighbors)
        if neighbors[0][1] == 0: # equal weighted average of neighbors with zero distance
            zero_distances = filter(lambda (k, v): v == 0, neighbors) # Retrieve tuples with zero distance
            prediction = float(sum(map(itemgetter(0), zero_distances))) / len(filter(lambda (k, v): v == 0, neighbors))
        else: # inverse distance weighted average of k nearest neighbors
            prediction = sum(map(lambda (x, y): y / float(x), zip(distances, responses))) / \
                         sum(map(lambda x: 1 / float(x), distances))
        print(prediction)











