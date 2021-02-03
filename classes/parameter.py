class Parameter(object):
    """
    function: 
        transfer the parameters in the functions.

    parameters: 
        None, but 'self'. 

    attributes:
        graph: class Graph, must, the graph data that you want to get the shortest path.
            (more info please see the developer documentation).
        BLOCK: tuple, a 3-tuple of integers as (x, y, z), the block size, to shape the kernal threads.
        GRID: tuple, a 2-tuple of integers as (x, y), the grid size, to shape the kernal blocks.
        useCUDA: bool, use CUDA to speedup or not.
        useMultiPro, bool, use multiprocessing in CPU or not. only support dijkstra APSP and MSSP.
        device: class, Device object. (see the 'SPoon/classes/device.py/Device') 
        srclist: list/int, the source of shortest path problem.
        sourceType: str, the type of the source.
        pathRecordBool: bool, record the path or not.
        part: int, the number of the edges that will put to GPU at a time.(divide algorithm)
    
    method:
        None, but init.
            
    return 
        class, Parameter object. (see the 'SPoon/classes/parameter.py/Parameter').
    """
    
    def __init__(self):

        # appoint the grid and block of the kernel
        self.BLOCK = None
        self.GRID = None

        self.device = None # devide parameter

        self.useCUDA = True # use CUDA or not
        self.useMultiPro = False # use multiple process or not

        self.graph = None # Graph Object
        self.srclist = None # source vertices 
        self.sourceType = None # SSSP APSP MSSP
        self.pathRecordBool = False # recording path or not

        self.part = None # batch size
        