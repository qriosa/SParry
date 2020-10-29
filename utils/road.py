def road(result, s, t):
	"""
	在某个单源问题中 求任意两个点之间的最短路径经过了哪些点
	实质就是遍历前驱进行查找
	接收 3个参数 result 类、 源点 s 和 终点 t
	返回 一个 list 即为依次经过哪些点
	"""
	assert result.path != None, "path记录为空，无法计算路径."



	return list