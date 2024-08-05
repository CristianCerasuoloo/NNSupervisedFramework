import logging

SEVERITY_MAPPING = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

class LoggerSingleton:
    _instance = None

    def __new__(cls, severity = "INFO", name='root'):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(name)
            cls._instance.logger.setLevel(SEVERITY_MAPPING[severity]) 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',  datefmt='%H:%M:%S')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            cls._instance.logger.addHandler(stream_handler)
        return cls._instance

    def set_level(self, level):
        self.logger.setLevel(level)

    def get_logger(self):
        return self.logger
    
# Utilizzo del Singleton per ottenere il logger
logger_singleton = LoggerSingleton("INFO")

def get_logger():
    return logger_singleton.get_logger()

def set_level(level):
    logger_singleton.set_level(level)