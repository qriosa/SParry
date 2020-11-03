from time import time

from classes.graph import Graph
from utils.debugger import Logger

logger = Logger(__name__) 

def read(filename = 'data.txt'):
    """
    function:
        read graph from file, and shape to a Graph object.
    
    parameters:
        filename: str, the graph data file name.
    
    return:
        class, Graph object.
    """

    logger.info("entering read func.")

    start = time()
    
    g = Graph(filename)

    end = time()

    g.msg += ('用时 t = %.3f s \n' % (end - start))

    return g
