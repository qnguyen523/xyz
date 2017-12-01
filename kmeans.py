import numpy as np
import pickle

def computeCentroid(data):
    # compute the centroid
    return np.mean(data, 0)


def computeSSE(data):
    # compute the SSE
    u = computeCentroid(data)
    return np.sum(np.linalg.norm(data - u, 2, 1))

def writeToFile_k_means(clusters):
    with open("k_means_format.dat", "w") as write_file:
        for i in clusters:
            write_file.write('{}\n'.format(int(i)))

def writeToFile_bisect(clusters):
    with open("bisect_format.dat", "w") as write_file:
        for i in clusters:
            write_file.write('{}\n'.format(int(i)))
#The K-means clustering algorithm
class KMeansClusterer:
    def writeToFile(self, cluster):
        with open("format.dat", "w") as write_file:
            write_file.write('{}'.format(cluster))

    def __init__(self, data=None, k=2, min_gain=1, max_iter=20,
                 max_epoch=1, verbose=True):
        """Learns from data if given."""
        if data is not None:
            print 'in __init__1',
            print k,min_gain,max_iter,max_epoch,verbose
            # self.fit(data, k, min_gain, max_iter, max_epoch, verbose)

        # data is an array of 1xn matrix
    def fit(self, data, k=2, min_gain=1, max_iter=20, max_epoch=1,
            verbose=True):
        clusters = []
        # Pre-process
        # convert data into matrix of 1xn matrix
        self.data = np.matrix(data)
        self.k = k
        self.min_gain = min_gain

        # initialize min_sse to infinity
        min_sse = np.inf

        for epoch in range(max_epoch):
            # Randomly initialize k centroids
            indices = np.random.choice(len(data), k, replace=False)
            # obtain k randoms matrix
            u = self.data[indices, :]
            # Loop
            t = 0
            # initialize min_sse to infinity
            old_sse = np.inf
            while True:
                t += 1
                # Cluster assignment
                # initilize k clusters to None
                centroids = [None] * k
                for index, x in enumerate(self.data):
                    # obtain index of minimum norm; j: index of minimum norm
                    # x: a matrix in self.data
                    # u: 7 randoms matrix in self.data
                    # j is the closest cluster of x
                    j = np.argmin(np.linalg.norm(x - u, 2, 1))
                    # print j
                # break
                    # group all matrixes that have the same j
                    if centroids[j] is None:
                        centroids[j] = x
                        clusters.append(j+1)
                        # print j+1
                        # print index, j, x
                    else:
                        # push x onto C[j]
                        centroids[j] = np.vstack((centroids[j], x))
                        clusters.append(j+1)
                # break
            # break
                # update centroids
                for j in range(k):
                    u[j] = computeCentroid(centroids[j])
                    # print u[j]
                # break

                # Loop termination condition
                if t >= max_iter:
                    break
                # total new sse for all j
                new_sse = np.sum([computeSSE(centroids[j]) for j in range(k)])
                # initial old_sse = infinity
                gain = old_sse - new_sse
                if verbose:
                    line = "Epoch {:2d} Iter {:2d}: SSE={:10.4f}, GAIN={:10.4f}"
                    print(line.format(epoch, t, new_sse, gain))
                # update values
                if gain < self.min_gain:
                    if new_sse < min_sse:
                        print 'updated'
                        min_sse, self.centroids, self.u, self.clusters = new_sse, centroids, u, clusters
                    break
                else:
                    old_sse = new_sse
                del clusters[:]

        # writeToFile_k_means(self.clusters)
        return self
# bisecting k-means clustering algorithm
class BisectingKMeansClusterer:
    def __init__(self, data, cols, max_k=7, min_gain=100, verbose=True):
        print 'in __init__2',
        print cols,min_gain,max_k,verbose
        if data is not None:
            self.fit(data, cols, max_k, min_gain, verbose)
    
    def fit(self, data, cols, max_k=7, min_gain=100, verbose=True):
        self.data = data
        print max_k, min_gain, verbose
        self.kmeans = KMeansClusterer()
        # make it an array with size = 1
        self.centroids= [data, ]
        self.k = len(self.centroids)
        # convert into 1x2 matrix; self.u
        self.u = np.reshape([computeCentroid(self.centroids[i]) for i in range(self.k)], (self.k, cols))    
        # return
        # numOfClusters = 1
        while True:
            # pick a cluster which has the biggest SSE
            # sse_list = [computeSSE(data) for data in self.centroids]
            # old_sse = np.sum(sse_list)
            # data = self.centroids.pop(np.argmax(sse_list))

            # pick a cluster which has the largest size
            data = self.centroids.pop(largestSize(self.centroids))
            # bisect it
            self.kmeans.fit(data, k=2, verbose=True)
            # add bisected clusters to our list
            self.centroids.append(self.kmeans.centroids[0])
            self.centroids.append(self.kmeans.centroids[1])
            #  self.k: number of current clusters
            self.k += 1
            # numOfClusters += 1
            # convert into array of 1 x cols matrix; self.u
            self.u = np.reshape(
                [computeCentroid(self.centroids[i]) for i in range(self.k)], (self.k, cols))
            # terminate when number of clusters = 7
            # print numOfClusters
            if self.k >= 7:
                break
        # listAllClustersToFile(self.C, self.data)
        storeToFile(self.centroids, self.data)
        return self

def largestSize(clusters):
    index_of_largest_size = 0
    largest_size = 0
    for i in xrange(len(clusters)):
        if len(clusters[i]) > largest_size:
            largest_size = len(clusters[i])
            index_of_largest_size = i
    return index_of_largest_size

def storeToFile(clusters, data):
    # store to files
    with open('self_C.dat', 'wb') as file:
        pickle.dump(clusters, file)

    with open('self_data.dat', 'wb') as file:
        pickle.dump(data, file)
    print 'done'
