import numpy as np

filename = 'VEW_data20000000_109000000.txt_nodirected.npz'
temp = np.load(filename)

V = temp['arr_0']
E = temp['arr_1']
W = temp['arr_2']

# filename = 'test.txt'
# from utils.readGraph import read
# g = read(filename)
# V, E, W = g.CSR[0], g.CSR[1], g.CSR[2]

from calc import calc

srclist = 0

r1 = calc(graph = (V, E, W), graphType = 'CSR', method = 'dij', useCUDA = True, srclist = srclist)
print(f"GPU finish, timeCost = {r1.timeCost}")

r2 = calc(graph = (V, E, W), graphType = 'CSR', method = 'dij', useCUDA = False, srclist = srclist)
print(f"CPU finish, timeCost = {r2.timeCost}")

from utils.check import check

with open(f'testDivideAns_{filename}.txt', 'w') as f:
	f.write(check(r1.dist, r2.dist)) # 检测两个答案是否相等
	f.write(f'\n\nGPU time cost = ' + str(r1.timeCost)) # 计算用时
	f.write(f'\n\nCPU time cost = ' + str(r2.timeCost)) # 计算用时

print(check(r1.dist, r2.dist))