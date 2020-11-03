import numpy as np
from time import time

from classes.result import Result

def mult(x):
    #z = []
    n = len(x)
    z = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            z[i,j] = np.min(x[i,:]+x[:,j])
                        
    return z

def matrix(matrix, n, pathRecordingBool = False):
    """
    function: use Floyd-Warshall algorithm in CPU to solve the APSP. 
        (more info please see the developer documentation) .
    
    parameters:  
        matrix: adjacency matrix of the graph data. (more info please see the developer documentation) .
        n: the number of the vertices in the graph.
        pathRecordingBool: record the path or not.
    
    return: Result(class).(more info please see the developer documentation) .  
    """
    t1 = time()

    p = np.array(matrix).astype(np.int32)
    m = 1
    while m < n - 1:
        p = mult(p)
        m = 2*m
    
    timeCost = time() - t1

    # 结果
    result = Result(dist = p, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(matrix = matrix)
    
    return result