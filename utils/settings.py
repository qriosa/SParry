# use debugger or not
debugger = True

# the INF
INF = 0x7f7f3f7f

# print different

def PRINT(chars = None):
	if chars == None:
		print("\033[0;36;40m" + '\n' + "\033[0m")
	else:
		print("\033[0;36;40m" + chars + "\033[0m")