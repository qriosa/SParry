# maybe no use
from time import time

from classes.graph import Graph
from utils.debugger import Logger

logger = Logger(__name__) 

def read(filename = 'data.txt', directed = False):
    """
    function:
        read graph from file, and shape to a Graph object.
    
    parameters:
        filename: str, the graph data file name.
    
    return:
        class, Graph object. (see the 'SPoon/classes/graph.py/Graph')
    """

    logger.info("entering read func.")

    start = time()
    
    g = Graph(filename, directed = directed)

    end = time()

    g.msg += ('用时 t = %.3f s \n' % (end - start))

    return g
