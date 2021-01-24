from pretreat import read
import numpy as np

if __name__ == '__main__':
	# 节点数的列表
	# ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]

	# 度的列表, 有一个度为 1 可以展示稀疏图
	# ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	ns = [800]
	ds = [16]
	for n in ns:
		for d in ds:
			m = n * d
			filename = f"./data/data_{n}_{m}"

			# read as CSV
			g_CSV = read(filename = filename)

			temp = np.load(f'./predata/data_{n}_{m}_CSV.npz')
			V, E, W = temp['arr_0'], temp['arr_1'],temp['arr_2']
			print((V == g_CSV.graph[0]).all())
			print((E == g_CSV.graph[1]).all())
			print((W == g_CSV.graph[2]).all())


			# read as edgeSet
			g_edge = read(filename = filename, method = 'edge')
			temp = np.load(f'./predata/data_{n}_{m}_edge.npz')
			V, E, W = temp['arr_0'], temp['arr_1'],temp['arr_2']
			print((V == g_edge.graph[0]).all())
			print((E == g_edge.graph[1]).all())
			print((W == g_edge.graph[2]).all())

			# 读取用这个
			# temp = np.load(filename)['arr_0']