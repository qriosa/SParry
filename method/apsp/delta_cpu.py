from time import time
import numpy as np

from classes.result import Result
from method.sssp.delta_cpu import delta_stepping as delta_sssp

def delta_stepping(CSR, n, delta, MAXN, pathRecordingBool = False):
    """
	function: use delta_stepping algorithm in CPU to solve the APSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertex in the graph.
        delta: the delta of this algorithm.
        MAXN: the max value of all the edge in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .
    """

    t1 = time()

    dist = []

    for s in range(n):
        resulti = delta_sssp(CSR, n, s, delta, MAXN, False)
        dist.append(resulti.dist)    
    
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result