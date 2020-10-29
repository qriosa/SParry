import hashlib
import numpy as np

def check(data1, data2, name1 = 'data1', name2 = 'data2'):
	m1 = hashlib.md5()
	m2 = hashlib.md5()

	m1.update(str(np.array(data1).flatten()).encode('utf-8'))
	m2.update(str(np.array(data2).flatten()).encode('utf-8'))

	if m1.hexdigest() == m2.hexdigest():
		return f"{name1} 和 {name2} 一样"
	else:
		return f"{name1} 和 {name2} 不一样"