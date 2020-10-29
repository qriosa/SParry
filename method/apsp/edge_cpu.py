from time import time
import numpy as np

from classes.result import Result
from method.sssp.edge_cpu import edge as edge_sssp

def edge(edgeSet, n, m, pathRecordingBool = False):
    """
	function: use edge free in CPU to solve the APSP. 
        (more info please see the developer documentation) .
	
	parameters:  
		edgeSet: edgeSet graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        m: the number of the edges in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .  
    """

    t1 = time()

    dist = []

    for s in range(n):
        resulti = edge_sssp(edgeSet, n, m, s, False)
        dist.append(resulti.dist)    
    
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(edgeSet = edgeSet)

    return result