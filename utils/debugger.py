from utils.settings import debugger

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Logger(object):
    """
    function: 
        set a logger.
    
    parameters: 
        name: str, the '__name__' of the caller.
    
    attributes:
        logger: class, a logging object.
    
    method:
        notset: the 'notset' of the logging.
        debug: the 'debug' of the logging.
        info: the 'info' of the logging.
        warning: the 'warning' of the logging.
        error: the 'error' of the logging.
        critical: the 'critical' of the logging.
    
    return: 
        class, Logger object.      
    """

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.debugger = debugger
    
    def notset(self, strings):
        """
        function: 
            the 'notset' of the logging.

        parameters:
            strings: str, what you want to notset.

        return 
            None, no return. 
        """
        if self.debugger == 0 or self.debugger > 1:
            return

        # 优先级为 1
        self.logger.info("notset: " + strings)

    def debug(self, strings):
        """
        function: 
            the 'debug' of the logging.

        parameters:
            strings: str, what you want to debug.

        return 
            None, no return. 
        """
        if self.debugger == 0 or self.debugger > 2:
            return
        
        # 优先级为  2
        self.logger.info("DEBUG: " + strings)

    def info(self, strings):
        """
        function: 
            the 'info' of the logging.

        parameters:
            strings: str, what you want to info.

        return 
            None, no return . 
        """
        if self.debugger == 0 or self.debugger > 3:
            return

        # 优先级为 3
        self.logger.info(strings)

    def warning(self, strings):
        """
        function: 
            the 'warning' of the logging.

        parameters:
            strings: str, what you want to warning.

        return 
            None, no return. 
        """
        if self.debugger == 0 or self.debugger > 4:
            return

        # 优先级为 4    
        self.logger.warning(strings)

    def error(self, strings):
        """
        function: 
            the 'error' of the logging.

        parameters:
            strings: str, what you want to error.

        return 
            None, no return. 
        """
        # 小于 6 就输出
        if self.debugger == 0 or self.debugger > 5:
            return 

        # 优先级为5
        self.logger.error(strings)

    def critical(self, strings):
        """
        function: 
            the 'critical' of the logging.

        parameters:
            strings: str, what you want to critical.

        return 
            None, no return. 
        """
        # 优先级为最高
        if self.debugger != 0:
            self.logger.critical(strings)
