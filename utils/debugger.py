import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Logger(object):
    """
    function: set a logger.
    
    parameters: 
        name: the '__name__' of the caller.
	
	attributes:
		logger: a logging object.
	
	method:
		notset: the 'notset' of the logging.
		debug: the 'debug' of the logging.
		info: the 'info' of the logging.
		warning: the 'warning' of the logging.
		error: the 'error' of the logging.
		critical: the 'critical' of the logging.
    
    return: Logger object.  	
    """

	def __init__(self, name):
	    self.logger = logging.getLogger(name)
    
    def notset(strings):
        """
		function: the 'notset' of the logging.

		parameters:
			strings: what you want to notset.

		return None. 
        """
        self.logger.notset(strings)

    def debug(strings):
        """
		function: the 'debug' of the logging.

		parameters:
			strings: what you want to debug.

		return None. 
        """
        self.logger.debug(strings)

    def info(strings):
        """
		function: the 'info' of the logging.

		parameters:
			strings: what you want to info.

		return None. 
        """
        self.logger.info(strings)

    def warning(strings):
        """
		function: the 'warning' of the logging.

		parameters:
			strings: what you want to warning.

		return None. 
        """
        self.logger.warning(strings)

    def error(strings):
        """
		function: the 'error' of the logging.

		parameters:
			strings: what you want to error.

		return None. 
        """
        self.logger.error(strings)

    def critical(strings):
        """
		function: the 'critical' of the logging.

		parameters:
			strings: what you want to critical.

		return None. 
        """
        self.logger.critical(strings)
