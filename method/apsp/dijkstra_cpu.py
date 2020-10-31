from time import time
import numpy as np

from classes.result import Result
from method.sssp.dijkstra_cpu import dijkstra as dij_sssp

def dijkstra(para):
    """
	function: use dijkstra algorithm in CPU to solve the APSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .
    """

    t1 = time()

    CSR, n, pathRecordingBool = para.CSR, para.n, para.pathRecordingBool

    dist = []

    for s in range(n):
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = None
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
