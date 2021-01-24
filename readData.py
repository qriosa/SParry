from pretreat import read
import numpy as np

if __name__ == '__main__':
	# 节点数的列表
	ns = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]

	# 度的列表, 有一个度为 1 可以展示稀疏图
	ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	for n in ns:
		for d in ds:
			m = n * d
			filename = f"./data/data_{n}_{m}"

			# read as CSV
			g_CSV = read(filename = filename)
			# 保存到 npz
			np.savez(f'./predata/data_{n}_{m}_CSV', g_CSV.graph[0], g_CSV.graph[1], g_CSV.graph[2])

			# read as edgeSet
			g_edge = read(filename = filename, method = 'edge')
			# 保存到 npz
			np.savez(f'./predata/data_{n}_{m}_edge', g_edge.graph[0], g_edge.graph[1], g_edge.graph[2])


			# 读取用这个
			# temp = np.load(filename)['arr_0']