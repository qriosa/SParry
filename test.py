from utils.debugger import Logger
from utils.settings import PRINT

import sys

# test update different file
def test(filename = None, n = 1000, m = 5000, l = 1, r = 20):
	'''
	developers use to  test.
	'''
	# set logging
	logger = Logger(__name__)

	if filename == None:
		# set the graph file name
		filename = 'test.txt'


		# import the generate func to generate a graph.
		from utils.genAGraph import generate 

		logger.info(f'func:generate, filename = {filename}, n = {n}, m = {m}')
		# generate a graph with n vertexs and m (undirected) edges, save it the the file
		generate(filename = filename, n = n, m = m, l = l, r = r) 



	# # import the read func to read a graph from file
	# from utils.readGraph import read 

	# logger.info(f'read filename = {filename}')
	# # read the graph
	# g = read(filename)

	# # print some infomation
	# print(g.msg)

	# n = g.n
	# m = g.m




	import numpy as np 

	logger.info(f'drive data to CSR')
	# # generate the data to CSR type
	# CSR = np.array([g.CSR[0], g.CSR[1], g.CSR[2]])
	# # print(CSR)




	# import the calc interface
	from calc import calc

	# show the help info
	# help(calc)



	# begin to calc 
	# method = ['dij']
	directed = True
	method = ['dij', 'delta', 'spfa', 'edge', 'dij', 'delta', 'spfa', 'edge']
	useCUDA = [True, True, True, True, False, False, False, False]
	# useCUDA = [False, False, False, False]
	useCUDATrue = 'useCUDA'
	useCUDAFalse = 'noUseCUDA'
	ans = []
	# srclist = [i for i in range(n)]
	# srclist = None
	# srclist = 232
	srclist = 2
	# srclist = [1,2,3,12,13,17]
	# srclist = [20,123,1114,5098,6111,9914,23,345,123,345,435,67,234,124,456,768,34,234,456,78,234,56,678,89,123,456,678,423,576,8964,6489,1999,2437,1031,5436,6522,1456,2345]
	# srclist = [2, 5, 12, 45, 45, 23, 87, 145, 567, 368, 325, 463, 168, 1276, 2416, 1567, 23, 4567, 2352, 3456, 2878, 2978, 1983]
	# srclist = [i for i in range(200, 1000, 40)]

	from utils.check import check 

	for i in range(len(method)):
		logger.info(f'begin to calc. method = {method[i]}, useCUDA = {useCUDA[i]}, pathRecordBool = False, directed = {directed}, srclist = {np.array(srclist)}')
		
		ans.append(calc(graph = filename, graphType = 'edgeSet', method = method[i], useCUDA = useCUDA[i], directed = directed, pathRecordBool = True, srclist = srclist, block=(512, 2, 1), grid = (1, 1)))
		
		logger.info(f'begin to check')
		
		PRINT(check(ans[0].dist, ans[i].dist, method[0], method[i])) # 检测两个答案是否相等
		PRINT(f'{method[i]}_{useCUDATrue if useCUDA[i] else useCUDAFalse} time cost = ' + str(ans[i].timeCost)) # 计算用时
		
		# print(ans[i].dist.reshape(len(srclist), n))
		# print(ans[i].dist)
		# print(ans[i].calcPath())
		# print(ans[i].path)
		# print(ans[i].display())
		# print(ans[i].drawPath())
		# input("enter enter to continue...")


if __name__ == '__main__':
	if len(sys.argv) == 2:
		filename = sys.argv[1]
	else:
		filename = None

	test(filename = filename, n = 1000, m = 2000, l = 3, r = 30)