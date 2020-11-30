# 利用服务器将大图转化为 numpy 进行测试

import numpy as np

def read(filename):

	try:
		with open(filename, 'r') as f:
			lines = f.readlines()
	except:
		filename = "./data/" + filename
		with open(filename, 'r') as f:
			lines = f.readlines()

	e = {}

	n = int(lines[0].split(' ')[0])
	m = int(lines[0].split(' ')[1])
	V = np.full((n + 1, ), 0).astype(np.int32)
	E = np.full((m, ), 0x7f7f3f7f).astype(np.int32)
	W = np.full((m, ), 0x7f7f3f7f).astype(np.int32)

	lines = lines[1:]

	for i in range(n):
		e[str(i)] = []

	for line in lines:
		line = (line[:-1]).split(' ')
		
		e[line[0]].append((int(line[1]), int(line[2])))
		# e[line[1]].append((int(line[0]), int(line[2]))) # 单向边

	last = 0

	vn = 0
	wn = 0

	for key in e:
		V[vn] = last
		vn += 1

		last += len(e[key])
		
		for j in e[key]:
			E[wn] = j[0]
			W[wn] = j[1]
			wn += 1

	V[vn] = last

	return n, m, V, E, W

if __name__ == '__main__':

	filenames = ['data10000_20000.txt', 'data4100000_82000000.txt', 'data6900000_101000000.txt', 'data18000000_109000000.txt', 'data20000000_109000000.txt', ]
	for filename in filenames:

		n, m, V, E, W = read(filename)
		np.savez(f'./numpydata/VEW_{filename}', V, E, W)
