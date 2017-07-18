from os import environ as E
from collections import namedtuple

ConfigClass = namedtuple('ConfigClass', 'ModelDir')

Conf = ConfigClass(
    E.get('MODEL_DIR','.'),
)
