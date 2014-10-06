""" Be careful! These paths are relative. """
from os import sys, path, sep

#def add_glrm_path():
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
data_dir = path.dirname(path.dirname(path.abspath(__file__))) + sep + "data"
