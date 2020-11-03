from time import time
import numpy as np

from method.sssp.delta_cpu import delta_stepping as delta_sssp
from classes.result import Result
from utils.debugger import Logger

logger = Logger(__name__)

def delta_stepping(para):
    """
    function: 
        use delta_stepping algorithm in CPU to solve the MSSP. 
    
    parameters:  
        class, Parameter object.
    
    return: 
        class, Result object. (more info please see the developer documentation) .
    """

    logger.info("turning to func delta_stepping-cpu-mssp")

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
