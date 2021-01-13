# test pretreat right?
# 测试生产图

from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.genAGraph import generate
from utils.check import check
from utils.settings import INF

from calc import calc
from pretreat import read

import numpy as np

def test(filename, srclists):

	methods = ['dij', 'spfa', 'delta']
	useCUDAs = [True, False]
	
	outname = [[], [], []]
	res = [[], [], []] # SSSP MSSP APSP

	g = read(filename = filename, detail = True)
	print(g.msg)

	print("=======>from file to CSR")
	for method in methods:
		for useCUDA in useCUDAs:
			for i in range(3):
				srclist = srclists[i]
				# 图都一样就修改方法就可以了 
				g.method = method
				res[i].append(calc(graph = g, useCUDA = useCUDA, srclist = srclist))
				outname[i].append(str("file->CSR: ")+str(method)+str(useCUDA)+str(np.array(srclist)))
	
	print("=======>from file to edge")
	# edge
	g = read(filename = filename, detail = True, method = "edge")
	for useCUDA in useCUDAs:
		for i in range(3):
			srclist = srclists[i]
			res[i].append(calc(graph = g, useCUDA = useCUDA, srclist = srclist))
			outname[i].append(str("file->edgeSet: ")+str(method)+str(useCUDA)+str(np.array(srclist)))

	print("=======>from edge to CSR")
	# 检验 edgeSet 转化 CSR 回去
	g1 = read(edgeSet = g.graph, detail = True)
	for method in methods:
		for useCUDA in useCUDAs:
			for i in range(3):
				srclist = srclists[i]
				# 图都一样就修改方法就可以了 
				g1.method = method
				res[i].append(calc(graph = g1, useCUDA = useCUDA, srclist = srclist))
				outname[i].append(str("edgeSet->CSR: ")+str(method)+str(useCUDA)+str(np.array(srclist)))	
	
	print("=======>from CSR to edge")
	# 检验 CSR 转化回 edgeSet
	g2 = read(CSR = g1.graph, method = 'edge', detail = True)
	for useCUDA in useCUDAs:
		for i in range(3):
			srclist = srclists[i]
			res[i].append(calc(graph = g2, useCUDA = useCUDA, srclist = srclist))
			outname[i].append(str("CSR->edgeSet: ")+str(method)+str(useCUDA)+str(np.array(srclist)))	


	# 检验矩阵能否行
	# 先从 CSR 生成矩阵
	matrix = np.full((g1.n, g1.n), INF, dtype=np.int32)
	for i in range(g1.n):
		for j in range(g1.graph[0][i], g1.graph[0][i+1]):
			matrix[i][g1.graph[1][j]] = min(matrix[i][g1.graph[1][j]], g1.graph[2][j])

	print("=======>from matrix to CSR")
	# 从矩阵检验 CSR
	g3 = read(matrix = matrix, detail = True)
	for method in methods:
		for useCUDA in useCUDAs:
			for i in range(3):
				srclist = srclists[i]
				# 图都一样就修改方法就可以了 
				g3.method = method
				res[i].append(calc(graph = g3, useCUDA = useCUDA, srclist = srclist))
				outname[i].append(str("matrix->CSR: ")+str(method)+str(useCUDA)+str(np.array(srclist)))	

	print("=======>from matrix to edge")
	# 从矩阵检验 edgeSet
	g4 = read(matrix = matrix, method = 'edge', detail = True)
	for useCUDA in useCUDAs:
		for i in range(3):
			srclist = srclists[i]
			res[i].append(calc(graph = g4, useCUDA = useCUDA, srclist = srclist))
			outname[i].append(str("matrix->edgeSet: ")+str(method)+str(useCUDA)+str(np.array(srclist)))	

	# check
	for j in range(3):
		print("\n")
		print("="*15 + str(np.array(srclists[j])) + "="*15) 
		for i in range(len(res[j])):
			print("="*15)
			check(res[j][0].dist, res[j][i].dist, outname[j][0], outname[j][i])

		# res[i].display()
		# res[i].drawPath()

if __name__ == "__main__":
	filename = f"./data/data10_20.txt"
	srclists = [56, [5,90,34,57,28], None]

	test(filename = filename, srclists = srclists)
