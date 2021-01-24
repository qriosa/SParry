# 通过图进行加速比测试

from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import checkBool

from calc import calc
from pretreat import read

import pandas as pd
from time import time
import numpy as np

# 写入CSV中
Ns = []
Ms = []
Methods = [] # 由于pd不支持非数字，所以用0123代表上面的方法
CheckCPUs = [] # 都和 dij 的串行相比较
CheckGPUs = []
TimeGPUs = []
TimeCPUs = []

def work(n, m, method):
	"""
	接收一个文件进行读取并进行用时测算
	"""

	if method != "edge":
		temp = np.load(f'./predata/data_{n}_{m}_CSV.npz')
		V, E, W = temp['arr_0'], temp['arr_1'],temp['arr_2']
		CSR = [V, E, W]
		g = read(CSR = CSR, method = method)
	else:
		temp = np.load(f'./predata/data_{n}_{m}_edge.npz')
		V, E, W = temp['arr_0'], temp['arr_1'],temp['arr_2']
		edgeSet = [V, E, W]
		g = read(edgeSet = edgeSet, method = method)

	

	# CPU 
	t1 = time()
	r = calc(graph = g, useCUDA = False, srclist = None)
	t2 = time()

	# res.append(r)
	# CheckCPUs.append(checkBool(res[0].dist, r.dist))
	# TimeCPUs.append((t2 - t1) * 100000 // 10 / 10000)

	return ((t2 - t1) * 100000 // 10 / 10000), r.dist



if __name__ == '__main__':
	
	# 节点数的列表
	ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
	# ns = [100]

	# 度的列表, 有一个度为 1 可以展示稀疏图
	ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
	# ds = [2]

	# CSV name 每次运行都是一个新的文件名
	filename = f"./testResult/apsp/test_CPU_{str(time())[11:]}.csv"

	methods = ['dij', 'spfa', 'delta', 'edge']

	old = []

	for n in ns:
		for d in ds:
			for method in methods:
				try:
					t, dist = work(n, n * d, method)

					Ns.append(n)
					Ms.append(n*d)
					Methods.append(method)
					TimeCPUs.append(t)

					# CPU 
					if method == "dij":
						CheckCPUs.append(True)
						old.append(dist)
					else:
						CheckCPUs.append(checkBool(old[0], dist))

					# GPU
					temp = np.load(f'./testResult/apsp/ans_GPU/dist_{n}_{n*d}_{method}.npz')
					dist_gpu = temp['arr_0']

					CheckGPUs.append(checkBool(old[0], dist_gpu))

				except:
					Ns.append(n)
					Ms.append(n*d)
					Methods.append(method)
					TimeCPUs.append("can not calc")
					CheckGPUs.append("can not calc")
					CheckCPUs.append("can not calc")
					
			
			# save
			df = pd.DataFrame({'n':Ns, 'm':Ms, 'method':Methods, 'checkGPU':CheckGPUs, 'timeCPU':TimeCPUs, 'checkCPU':CheckCPUs})
			df.to_csv(filename)
			print(f"saved as {filename}")