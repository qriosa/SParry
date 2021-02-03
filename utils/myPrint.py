# print red
def PRINT_red(chars = None, ptype = "\033[0;31;40m"):
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

# print blue
def PRINT_blue(chars = None, ptype = "\033[0;36;40m"):
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