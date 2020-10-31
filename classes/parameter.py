class Parameter(object):
	"""
	function: to transfer the parameters in the functions.

	parameters: None.

	attributes:
		n: the number of the vertexs in the graph.
		m: the number of the edges in the graph.
		useCUDA: use CUDA to speedup or not.
		CSR: CSR of graph data.
		matrix: adjacency matrix of graph data.
		edgeSet: collection of edges.
		graphType: type of graph.
		method: the algorithm.
		srclist: the source of shortest path problem.
		sourceType: the type of the problem. [APSP, SSSP, MSSP]
		pathRecordingBool: record the path or not.
		delta: the delta of delta-stepping algorithm.
		maxOutDegree: the max out degree of the graph.
		part: the number of the edges that will put to GPU at a time.(divide algorithm)
		streamNum: the number of streams used.
			
	return parameter object.
	"""
	def __init__(self):

		self.BLOCK = None
		self.GRID = None
		
		self.n = None # 结点数量
		self.m = None # 边的数量（为了兼容有向边的数量，无向边应自动乘2）
		self.directed = None # 指定图是否有向
		self.valueType = None # 边权数据类型 int float

		# 指定kernel使用的grid block
		self.grid = None
		self.block = None

		self.useCUDA = True # 是否使用 CUDA

		self.CSR = None # 压缩邻接矩阵
		self.matrix = None # 邻接矩阵
		self.edgeSet = None # 边  (src, des, w)
		self.graphType = None # 传入的图的类型 

		self.method = None # 使用的计算方法(dij\spfa\delta\edge\matrix)
		self.filepath = None # 读取图的文件路径
		self.srclist = None # 源点的集合 单个源点的[数字编号]、全源的[无]、多源的[list] 
		self.sourceType = None # SSSP APSP MSSP

		self.pathRecordingBool = False # 是否记录路径
		self.output = None # 输出结果的文件路径（默认生成三个文件？dist、path？）
		self.logBool = None # 是否打印调试日志

		# delta 还需要一些参数
		self.delta = None
		self.maxOutDegree = None # 最大出度 
		self.MAXN = -1

		# 以下是分块的参数
		self.part = None # 分块中一次拷贝的边的数目
		self.sNum = None # 多源中一次解决多少个问题

		# 以下是多流的属性参数
		self.streamNum = None # 指定流的数量
		self.blockSize = None # 多流中一块的分块的边的数量

		# 以下是矩阵相乘
		self.blockNum = None # 矩阵分块的块数


		