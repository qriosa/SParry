from time import time
import numpy as np

from method.sssp.delta_cpu import delta_stepping as delta_sssp
from classes.result import Result

def delta_stepping(CSR, n, srclist, delta, MAXN, pathRecordingBool = False):
    """
	function: use delta_stepping algorithm in CPU to solve the MSSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        srclist: the source list, can be list.(more info please see the developer documentation).
        delta: the delta of this algorithm.
        MAXN: the max value of all the edges in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .   
    """

    t1 = time()

    dist = []

    for s in srclist:
        resulti = delta_sssp(CSR, n, s, delta, MAXN, False)
        dist.append(resulti.dist)    
    
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)
            
    return result
