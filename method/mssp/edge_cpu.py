from time import time
import numpy as np

from classes.result import Result
from method.sssp.edge_cpu import edge as edge_sssp

def edge(edgeSet, n, m, srclist, pathRecordingBool = False):
    """
	function: use edge free in CPU to solve the MSSP. 
        (more info please see the developer documentation) .
	
	parameters:  
		edgeSet: edgeSet graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        m: the number of the edges in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .   
    """

    t1 = time()

    dist = []

    for s in srclist:
        resulti = edge_sssp(edgeSet, n, m, s, pathRecordingBool)
        dist.append(resulti.dist)    
    
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(edgeSet = edgeSet)

    return result