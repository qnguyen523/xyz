import pickle
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

def listAllClustersToFile(clusters, data):
    print 'len(clusters)', len(clusters)
    with open("list_bi_clusters.dat", "w") as write_file:
        for i in xrange(len(clusters)):
            print i, len(clusters[i])
            write_file.write('{} {}\n'.format(i, len(clusters[i])))
    
    # print data[0]
    # print clusters[0][1]
    # print (data[0] == clusters[0][1])
    # print (np.all(data[0] == clusters[0][1]))

    # with open("format.dat", "w") as write_file:
    #     for x in data:
    #         flag = False
    #         for i in xrange (len(clusters)):
    #             for y in clusters[i]:
    #                 if np.all(x == y):
    #                     flag = True
    #                     write_file.write('{}\n'.format(i+1))    
    #                     break
    #             if flag == True:
    #                 break
    #         if flag == False:
    #             print 'error'
            


                


                # if np.all(x in clusters[i]):
                #     print i
                #     write_file.write('{}\n'.format(i+1))
                #     break

    # print len(clusters)
    # print len(data)

    with open("format.dat", "w") as write_file:
        t = 0
        for h in xrange(len(data)):
            print t
            flag = False
            for i in xrange(len(clusters)):
                for j in xrange(len(clusters[i])):
                    if np.all(data[h] == clusters[i][j]):
                        flag = True
                        write_file.write('{}\n'.format(i+1))
                        break
                if(flag == True):
                    break
            if(flag == False):
                print 'error'
                write_file.write('{}\n'.format(1))
            t += 1


# main
# load clusters
with open('self_C.dat', 'rb') as file:
    clusters = pickle.load(file)
# print clusters[0]
# load data
with open('self_data.dat', 'rb') as file:
    data = pickle.load(file)

data = np.matrix(data)
# data = data.toarray()
# clusters = np.matrix(clusters)

# final_clusters=[]
# for i,x in enumerate(clusters):
#     temp_clusters = []
#     for j,y in enumerate(clusters[i]):
#         temp_clusters.append(y.nonzero()[1])
#     final_clusters.append(temp_clusters)
#     # del temp_clusters[:]

# final_data=[]
# data = np.matrix(data)
# for x in data:
#     final_data.append(x.nonzero()[1])

listAllClustersToFile(clusters ,data)
