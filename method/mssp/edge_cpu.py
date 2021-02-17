from time import time
import numpy as np

from classes.result import Result
from method.sssp.edge_cpu import edge as edge_sssp
from utils.debugger import Logger

logger = Logger(__name__)

def edge(para):
    """
    function: 
        use edgeSet in CPU to solve the MSSP.  (more info please see the developer documentation).
    
    parameters:  
        class, Parameter object. (see the 'sparry/classes/parameter.py/Parameter').
    
    return: 
        class, Result object. (see the 'sparry/classes/result.py/Result').
    """

    logger.debug("turning to func edge-cpu-mssp")

    t1 = time()

    srclist, pathRecordBool = para.srclist.copy(), para.pathRecordBool

    dist = []

    for s in srclist:
        para.srclist = s
        resulti = edge_sssp(para)
        dist.append(resulti.dist)    
    
    para.srclist = srclist
    dist = np.array(dist)

    timeCost = time() - t1

    # result
    result = Result(dist = dist, timeCost = timeCost, graph = para.graph)

    if pathRecordBool:
        result.calcPath()

    return result