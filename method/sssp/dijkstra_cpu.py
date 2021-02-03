from queue import PriorityQueue
# from threading import Thread, Lock
from time import time
import numpy as np

from utils.settings import INF
from utils.debugger import Logger
from classes.result import Result

logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter'). 
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').    
    """

    logger.debug("turning to func dijkstra-cpu-sssp")

    return dij_serial(para)


def dij_serial(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the SSSP. 
    
    parameters:  
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'SPoon/classes/result.py/Result').
    """

    # logger.debug("turning to func dij_serial-sssp").

    t1 = time()

    CSR, n, s, pathRecordBool = para.graph.graph, para.graph.n, para.srclist, para.pathRecordBool

    V, E, W = CSR[0], CSR[1], CSR[2]

    # priority queue
    q = PriorityQueue()

    # dist
    dist = np.full((n,), INF).astype(np.int32)
    dist[s] = 0

    # vis 
    vis = np.full((n, ), 0).astype(np.int32)

    q.put((0, s)) # put the source vertex into priority queue

    while q.empty() == False:
        p = q.get()[1]

        if vis[p] == 1: 
            continue

        vis[p] = 1

        for j in range(V[p], V[p + 1]):
            if dist[E[j]] > dist[p] + W[j]:
                dist[E[j]] = dist[p] + W[j]
                q.put((dist[E[j]], E[j]))

    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result

