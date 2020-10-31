import numpy as np
from utils import settings

# 0 号结点开始编号

def getIndex(para):
	"""
	function: get n(the number of the graph's vertex), m(the number of the edge in the graph),
		and if the algorithm is 'delta_stepping' it will also get the delta value. 
	
	parameters:
		a parameters class. (more info please see the developer documentation) . 
	
	return: None.
	"""
	if para.graphType == 'matrix':

		para.matrix = np.array(para.matrix,dtype=np.int32)
		assert para.matrix.shape[0] == para.matrix.shape[1], "需要一个 (n, n) 的矩阵，但接收到 " + str(para.matrix.shape)
		para.n = np.int32(para.matrix.shape[0]) # 这里应该是不需要 + 1 的
		
		# matrix[i][j] > 0 意味着存在从 i -> j 的边
		para.m = 0
		para.MAXN = 0
		para.maxOutDegree = 0
		for i in range(para.n):
			temp = 0 # 统计 i 的出度 
			for j in range(para.n):
				if para.matrix[i][j] < INF:
					para.m += 1
					temp += 1
					para.MAXN = max(para.MAXN, para.matrix[i][j])

			para.maxOutDegree = max(para.maxOutDegree, temp)
		
		para.m = np.int32(para.m)
		para.MAXN = np.int32(para.MAXN)
		para.maxOutDegree = np.int32(para.maxOutDegree)

	# 只计算 n m src des w
	elif para.graphType == 'edgeSet':
		# 以最大的结点来判定结点的个数 n = maxId + 1 因为统一考虑有0号点 端点保证左闭右开
		para.n = 0
		
		# 全是有向边
		# if para.directed == True:
		para.m = np.int32(len(para.edgeSet[0]))

		para.n = max(para.edgeSet[0].max(), para.edgeSet[1].max()) +1

		
		# else:
		# 	para.m = np.int32(len(para.edgeSet[0]))

		# 	para.src = para.edgeSet[0] 
		# 	para.des = para.edgeSet[1]
		# 	para.w = para.edgeSet[2]
		# 	para.n = max(para.src.max(), para.des.max()) +1


	elif para.graphType == 'CSR' or para.graphType == None:

		# 5 项都不为空就没有求的必要了 目前仅仅只求 5 样
		# if para.n != None and para.m != None and para.delta != None and para.MAXN != None and para.maxOutDegree != None:
		# 	return

		# V, E, W = list(para.CSR[0]), para.CSR[1], para.CSR[2]
		V, E, W = para.CSR[0], para.CSR[1], para.CSR[2]

		para.CSR[0] = np.array(V, dtype = np.int32)
		para.CSR[1] = np.array(para.CSR[1], dtype = np.int32)
		para.CSR[2] = np.array(para.CSR[2], dtype = np.int32)

		para.n = np.int32(len(V) - 1) # 这里 +1 呢 保证端点是左闭右开
		para.m = np.int32(len(E))
		
		para.MAXN = -1
		for w in W:
			if w > para.MAXN:
				para.MAXN = w
		para.MAXN = np.int32(para.MAXN)
		
		para.maxOutDegree = 0
		for i in range(1, para.n):
			para.maxOutDegree = max(para.maxOutDegree, V[i] - V[i - 1])
			# if V[i] - V[i - 1] > para.maxOutDegree:
			# 	para.maxOutDegree = V[i] - V[i - 1]
		para.maxOutDegree = np.int32(para.maxOutDegree)
		
		if para.delta == None:
			para.delta = max(3, para.MAXN // para.maxOutDegree)
		
		para.delta = np.int32(para.delta)
	
	else:
		raise Exception("can not extract indexs from a graph with an undefined type")
		