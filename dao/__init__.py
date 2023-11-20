import os
import sys

pt_dao_dir = os.getcwd() + '/build/lib'
sys.path.append(pt_dao_dir)

from pt_dao import *

del os
del sys