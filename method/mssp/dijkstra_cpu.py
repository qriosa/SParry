from time import time
import numpy as np

from method.sssp.dijkstra_cpu import dijkstra as dij_sssp
from classes.result import Result

def dijkstra(para):
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

    CSR, n, srclist, pathRecordingBool = para.CSR, para.n, para.srclist.copy(), para.pathRecordingBool

    dist = []

    for s in srclist:
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = srclist
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)

    return result
