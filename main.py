import pickle
from kmeans import KMeansClusterer, BisectingKMeansClusterer

def loadData():
    with open('data.dat', 'rb') as file:
        data = pickle.load(file)
    data = data.toarray()
    # print data
    return data

def mainKMeansClusterer(data):
    k=7
    min_gain=100
    max_iter=10
    epoch=1
    verbose=True
    # initialize clusterer
    c = KMeansClusterer(data, k, min_gain, max_iter, max_epoch,verbose)

def mainBisectingKMeansClusterer(data, cols):
    max_k = 7
    min_gain = 100
    verbose = True
    # initialize clusterer
    c = BisectingKMeansClusterer(data, cols, max_k, min_gain, verbose)

if __name__ == '__main__':
    # clustering
    # mainKMeansClusterer(loadData())
    mainBisectingKMeansClusterer(loadData(), loadData().shape[1])