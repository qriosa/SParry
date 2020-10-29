from time import time
import numpy as np

from classes.result import Result
from method.sssp.spfa_cpu import spfa as spfa_sssp



def spfa(CSR,n,pathRecordingBool):
    """
	function: use Bellman-Ford algorithm in CPU to solve the APSP. 
	
	parameters:  
		CSR: CSR graph data. (more info please see the developer documentation) .
        n: the number of the vertexs in the graph.
        pathRecordingBool: record the path or not.
	
	return: Result(class).(more info please see the developer documentation) .    
    """
    global Va,Ea,Wa
    start_time = time()
    Va=CSR[0]
    Ea=CSR[1]
    Wa=CSR[2]
    dist=[]
    for st in range(n):
        resi = spfa_sssp(CSR,n,st)
        dist.append(resi.dist)
    
    dist = np.array(dist)
    end_time = time()
    timeCost = end_time - start_time
    result = Result(dist = dist, timeCost = timeCost)
    return result
