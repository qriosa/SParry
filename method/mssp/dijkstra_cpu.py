from time import time
import numpy as np

from method.sssp.dijkstra_cpu import dijkstra as dij_sssp
from classes.result import Result
from utils.debugger import Logger

logger = Logger(__name__)

def dijkstra(para):
    """
    function: 
        use dijkstra algorithm in CPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func dijkstra-cpu-mssp")

    t1 = time()

    srclist, pathRecordBool = para.srclist.copy(), para.pathRecordBool

    dist = []

    for s in srclist:
        para.srclist = s
        resulti = dij_sssp(para) 
        dist.append(resulti.dist)    
    
    para.srclist = srclist
    dist = np.array(dist)

    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result
