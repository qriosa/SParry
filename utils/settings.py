# use debugger or not
debugger = True

# the INF
INF = 0x7f7f3f7f

# force divide debug
# 如果开启 则会强制进入分图路由 调试所用
forceDivide = False


# print different
def PRINT(chars = None):
	if chars == None:
		print("\033[0;36;40m" + '\n' + "\033[0m")
	else:
		print("\033[0;36;40m" + chars + "\033[0m")