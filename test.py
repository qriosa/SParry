import logging

def test_delta_cpu_sssp(filename = None, n = 1000, m = 5000, l = 1, r = 20):
	'''
	this test for delta_stepping CPU SSSP arong.
	'''
	# set logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# logger = logging.getLogger(__name__)

	if filename == None:
		# set the graph file name
		filename = 'test.txt'


		# import the generate func to generate a graph.
		from genAGraph import generate 

		# logger.info(f'generate filename = {filename}, n = {n}, m = {m}')
		# generate a graph with n vertexs and m (undirected) edges, save it the the file
		generate(filename = filename, n = n, m = m, l = l, r = r) 



	# import the read func to read a graph from file
	from readGraph import read 

	# logger.info(f'read filename = {filename}')
	# read the graph
	g = read(filename)

	# print some infomation
	print(g.msg)




	import numpy as np 

	# logger.info(f'drive data to CSR')
	# generate the data to CSR type
	CSR = np.array([g.V, g.E, g.W])
	# print(CSR)




	# import the calc interface
	from calc import main

	# show the help info
	help(main)



	# begin to calc 
	method1, method2, method3 = 'dij', 'delta', 'delta'
	useCUDA1, useCUDA2, useCUDA3 = True, True, True
	srclist = 5

	# logger.info(f'begin to calc. method = {method1}, useCUDA = {useCUDA1}, pathRecordBool = False, srclist = 1')

	# check the dijkstra and delta_stepping is same or not.
	r1 = main(graph = CSR, graphType = 'CSR', method = method1, useCUDA = useCUDA1, pathRecordBool = False, srclist = srclist) 
	r2 = main(graph = CSR, graphType = 'CSR', method = method2, useCUDA = useCUDA2, pathRecordBool = False, srclist = srclist) 



	# logger.info(f'begin to check')
	# import the check ans to check the ans.
	from check import check 
	print(check(r1.dist, r2.dist, method1, method2)) # 检测两个答案是否相等
	
	useCUDATrue = 'useCUDA'
	useCUDAFalse = 'noUseCUDA'

	print(f'{method1}_{useCUDATrue if useCUDA1 else useCUDAFalse} time cost = ', r1.timeCost) # 并行计算用时
	print(f'{method2}_{useCUDATrue if useCUDA2 else useCUDAFalse} time cost = ', r2.timeCost) # 串行计算用时
	# print("speedup: " + str(r2.timeCostNum / (r1.timeCostNum * 1000)  * 100000 // 100)) # 显示两者的用时加速比

	r3 = main(graph = CSR, graphType = 'CSR', method = method3, useCUDA = useCUDA3, pathRecordBool = False, srclist = srclist) 
	print(check(r1.dist, r3.dist, method1, method3)) # 检测两个答案是否相等
	print(f'{method3}_{useCUDATrue if useCUDA3 else useCUDAFalse} time cost = ', r3.timeCost) # 串行计算用时

if __name__ == '__main__':
	test_delta_cpu_sssp(n = 80000, m = 954000, l = 3, r = 100)