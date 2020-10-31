import numpy as np

from utils.settings import INF
from utils.debugger import Logger

# set logging
logger = Logger(__name__)


class Graph(object):
	"""
	function: 
		a graph class.

	parameters:
		filename: the graph data file. (more info please see the developer documentation).
	
	attributes:
		n: the number of the vertexs in the graph.
		m: the number of the edges in the graph.
		CSR: CSR of graph data.
		src, des, w: the edge set triples.
		MAXW: the max weight of the edges.
		MINW: the min weight of the edges.
		MAXD: the max degree(In degree and Out degree) of all the vertexs.
		MAXU: one of the vertexs with the max degree.
		MIND: the min degree(In degree and Out degree) of all the vertexs.
		MINU: one of the vertexs with the min degree.
		degree: a list, save the degree of each vertex.
		msg: the message about the read func.
	
	method:
		read: read the graph from file.
		reshape: convert data to numpy.int32.

	return: 
		Graph object.
	"""

	# 只可以接受一个delta 和 s
	def __init__(self, filename = None):

		# 预定义变量
		self.n = -1
		self.m = -1

		self.CSR = None

		self.src = []
		self.des = []
		self.w = []
		self.edgeSet = []

		self.MAXW = -1 # 最大边权
		self.MINW = INF # 最小边权
		self.MAXD = -1 # 最大度
		self.MAXU = -1 # 最大度的点(之一)
		self.MIND = INF # 最小度
		self.MINU = -1 # 最小度的点(之一)

		self.degree = None

		self.msg = '欢迎使用'
		
		if filename != None:
			self.read(filename)
			self.reshape()


	def read(self, filename):
		"""
		function: 
			read the graph from file.
			only accept graphic data in edgeSet format and store it in memory in CSR/edgeSet format
			by the way, we wanna to specify the edgeSet format as a variable with 3 array src/des/weight which are consistent with every edge in graph

		parameters: 
			filename : the graph data file. (more info please see the developer documentation).

		return:
			None.
		"""

		logger.info(f"reading graph from {filename}...")

		try:
			with open(filename, 'r') as f:
				lines = f.readlines()
		except:
			filename = "./data/" + filename
			try:
				with open(filename, 'r') as f:
					lines = f.readlines()
			except:
				raise Exception("No such a file!")
		
		self.CSR = [[], [], []]

		e = {}

		self.n = int(lines[0].split(' ')[0])
		self.m = int(lines[0].split(' ')[1])
		self.degree = np.full((self.n, ), 0).astype(np.int32)

		lines = lines[1:]

		for i in range(self.n):
			e[str(i)] = []

		for line in lines:
			line = (line[:-1]).split(' ')
			
			e[line[0]].append((int(line[1]), int(line[2])))
			e[line[1]].append((int(line[0]), int(line[2])))
			
			self.src.append(int(line[0]))
			self.des.append(int(line[1]))
			self.w.append(int(line[2]))

			self.src.append(int(line[1]))
			self.des.append(int(line[0]))
			self.w.append(int(line[2]))

			if int(line[2]) > self.MAXW:
				self.MAXW = int(line[2])

			if int(line[2]) < self.MINW:
				self.MINW = int(line[2])


			self.degree[self.src[-1]] += 1
			self.degree[self.des[-1]] += 1

		last = 0
		for key in e:
			self.CSR[0].append(last)
			last += len(e[key])
			
			if len(e[key]) > self.MAXD:
				self.MAXD = len(e[key])
				self.MAXU = key

			if len(e[key]) < self.MIND:
				self.MIND = len(e[key])
				self.MINU = key

			for j in e[key]:
				self.CSR[1].append(j[0])
				self.CSR[2].append(j[1])

		self.CSR[0].append(last)

	def reshape(self):
		
		"""
		function: 
			convert data to numpy.int32.

		parameters: 
			None.

		return:
			None.
		"""

		logger.info(f"converting the graph to numpy.int32...")

		self.n = np.int32(self.n) # 结点数量
		self.m = np.int32(self.m) # 边的数量

		self.CSR[0] = np.copy(self.CSR[0]).astype(np.int32) # CSR的V
		self.CSR[1] = np.copy(self.CSR[1]).astype(np.int32) # CSR的E
		self.CSR[2] = np.copy(self.CSR[2]).astype(np.int32) # CSR的W

		self.src = np.copy(self.src).astype(np.int32) # 每个边的起点
		self.des = np.copy(self.des).astype(np.int32) # 每个边的终点
		self.w = np.copy(self.w).astype(np.int32) # 每个边的边权
		self.edgeSet = [self.src, self.des, self.w]

		self.MAXW = np.int32(self.MAXW) # 最大边权
		self.MINW = np.int32(self.MINW) # 最小边权
		self.MAXD = np.int32(self.MAXD) # 最大度
		self.MAXU = np.int32(self.MAXU) # 最大度的点(之一)
		self.MIND = np.int32(self.MIND) # 最小度
		self.MINU = np.int32(self.MINU) # 最小度的点(之一)

		self.msg = f"读取完毕:\n结点数量 n = {self.n}\n无向边数量 m = {self.m}\n最大边权 MAXW = {self.MAXW}\n最大度 degree({self.MAXU}) = {self.MAXD}\n最小度 degree({self.MINU}) = {self.MIND}\n"

