import logging

logger = logging.getLogger('mbpoker')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s; %(name)s; %(levelname)s]: %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)
