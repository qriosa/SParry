from time import time
import numpy as np

from method.sssp.dijkstra_cpu import dijkstra as dij_sssp
from classes.result import Result

def dijkstra(CSR, n, srclist, pathRecordingBool = False):
    """
	function: use dijkstra algorithm in CPU to solve the MSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .  
    """

    t1 = time()

    dist = []

    for s in srclist:
        resulti = dij_sssp(CSR, n, s, False) 
        dist.append(resulti.dist)    
    
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
