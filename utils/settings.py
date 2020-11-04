# use debugger or not
debugger = True

# the INF
INF = 0x7f7f3f7f

# force divide debug
# 如果开启 则会强制进入分图路由 调试所用
forceDivide = False


# print different
def PRINT(chars = None, ptype = "\033[0;36;40m"):
    """
    function:
        print different.
    
    parameters:
        chars: str, the content you can to print.
        ptyte: str, the type of print.
    
    return:
        None, no return.    
    """
    
    if chars == None:
        print(ptype + '\n' + "\033[0m")
    else:
        print(ptype + chars + "\033[0m")