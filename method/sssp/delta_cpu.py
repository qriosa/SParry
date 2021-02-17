from time import time
import numpy as np

from utils.settings import INF
from utils.debugger import Logger
from classes.result import Result

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result'). 
    """

    logger.debug("turning to func delta_stepping-cpu-sssp")

    # start time
    t1 = time()

    CSR, n, s, delta, pathRecordBool = para.graph.graph, para.graph.n, para.srclist, para.graph.delta, para.pathRecordBool

    V, E, W = CSR[0], CSR[1], CSR[2]

    # the max weight
    if para.graph.MAXW <= 0:
        MAXN = max(W)
    else:
        MAXN = para.graph.MAXW
    
    # dist
    dist = np.full((n, ), INF).astype(np.int32)
    dist[s] = 0   

    B = []

    # init
    isin = np.full((n, ), -1).astype(np.int32)

    maxidB = MAXN * (n - 2) // delta + 1 # calc the maxid of bucket

    for i in range(maxidB):
        B.append([])

    B[0].append(s) # put source into bucket 0
    isin[s] = 0


    idB = 0 # current id start from 0
    while True:
        
        tag = 0 

        for i in range(idB, maxidB):
            if B[i] != []: # find the first bucket that is not empty
                tag = 1
                idB = i # goto the first bucket that is not empty quickly
                break

        if tag == 0: # all bucket is empty
            break
        
        head = 0 # the number of vertices that have been used
        tail = len(B[idB]) # the number of vertex in the bucket

        while head < tail:

            p = B[idB][head]

            isin[p] = -1 

            for i in range(V[p], V[p + 1]):
                if W[i] <= delta:
                    if dist[E[i]] > dist[p] + W[i]:

                        dist[E[i]] = dist[p] + W[i]
                        newId = dist[E[i]] // delta

                        if isin[E[i]] != -1 and isin[E[i]] != newId:
                            B[isin[E[i]]].remove(E[i])
                        
                        if isin[E[i]] != newId:
                            isin[E[i]] = newId
                            
                            B[isin[E[i]]].append(E[i])

                            if newId == idB:
                                tail += 1

            head += 1

        for p in B[idB]:

            for i in range(V[p], V[p + 1]):
                if W[i] > delta:
                    if dist[E[i]] > dist[p] + W[i]:

                        dist[E[i]] = dist[p] + W[i]
                        newId = dist[E[i]] // delta

                        if isin[E[i]] != -1 and isin[E[i]] != newId:
                            B[isin[E[i]]].remove(E[i])
                        
                        
                        isin[E[i]] = newId
                        
                        B[isin[E[i]]].append(E[i])
                            
        idB += 1 # the next bucket

    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
