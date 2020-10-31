from time import time
import numpy as np

from method.sssp.delta_cpu import delta_stepping as delta_sssp
from classes.result import Result

def delta_stepping(para):
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

    CSR, n, srclist, delta, MAXN, pathRecordingBool = para.CSR, para.n, para.srclist.copy(), para.delta, para.MAXN, para.pathRecordingBool

    dist = []

    for s in srclist:
        para.srclist = s
        resulti = delta_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = srclist
    dist = np.array(dist)

    timeCost = time() - t1

    # 结果
    result = Result(dist = dist, timeCost = timeCost)

    if pathRecordingBool:
        result.calcPath(CSR = CSR)
            
    return result
