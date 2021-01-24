# 通过图进行加速比测试

from utils.myPrint import PRINT_blue
from utils.myPrint import PRINT_red
from utils.check import checkBool

from time import time
import pandas as pd
from random import randint

from calc import calc
from pretreat import read


# 写入CSV中
Ns = []
Ms = []
Methods = [] # 由于pd不支持非数字，所以用0123代表上面的方法
CheckCPUs = [] # 都和 dij 的串行相比较
CheckGPUs = []
TimeGPUs = []
TimeCPUs = []

def work(n, m):
	"""
	接收一个文件进行读取并进行用时测算
	"""
	
	filename = f"./data/data_{n}_{m}"
	g = read(filename = filename)


	methods = ['dij', 'spfa', 'delta']
	res = []

	# GPU  存在一个启动慢问题 所以先启动了
	r = calc(graph = g, useCUDA = True, srclist = 2)

	# 随机源点
	s = randint(0, n-2)

	for method in methods:
		g.method = method

		Ns.append(n)
		Ms.append(m)
		Methods.append(method)	

		# CPU 
		t1 = time()
		r = calc(graph = g, useCUDA = False, srclist = s)
		t2 = time()

		res.append(r)
		CheckCPUs.append(checkBool(res[0].dist, r.dist))
		TimeCPUs.append((t2 - t1) * 100000 // 10 / 10000)
		
		# GPU 
		t1 = time()
		r = calc(graph = g, useCUDA = True, srclist = s)
		t2 = time()

		res.append(r)
		CheckGPUs.append(checkBool(res[0].dist, r.dist))
		TimeGPUs.append((t2 - t1) * 100000 // 10 / 10000)


	# edge
	g = read(filename = filename, method = 'edge')

	Ns.append(n)
	Ms.append(m)
	Methods.append('edge')	

	# CPU 
	t1 = time()
	r = calc(graph = g, useCUDA = False, srclist = s)
	t2 = time()

	res.append(r)
	CheckCPUs.append(checkBool(res[0].dist, r.dist))
	TimeCPUs.append((t2 - t1) * 100000 // 10 / 10000)
	
	# GPU 
	t1 = time()
	r = calc(graph = g, useCUDA = True, srclist = s)
	t2 = time()

	res.append(r)
	CheckGPUs.append(checkBool(res[0].dist, r.dist))
	TimeGPUs.append((t2 - t1) * 100000 // 10 / 10000)


	print(res[0].dist)
	print(res[1].dist)
	print((res[0].dist == res[1].dist).all())


if __name__ == '__main__':
	
	# 节点数的列表
	ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]

	# 度的列表, 有一个度为 1 可以展示稀疏图
	ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	# CSV name 每次运行都是一个新的文件名
	filename = f"./testResult/sssp/test_{str(time())[11:]}.csv"


	for n in ns:
		for d in ds:
			work(n, n * d)
			# save
			df = pd.DataFrame({'n':Ns, 'm':Ms, 'method':Methods, 'timeGPU':TimeGPUs, 'checkGPU':CheckGPUs, 'timeCPU':TimeCPUs, 'checkCPU':CheckCPUs})
			df.to_csv(filename)