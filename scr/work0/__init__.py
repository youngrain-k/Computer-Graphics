# __init__.py
__version__ = "1.0.0"

# 导出外部可直接导入的函数/变量
from .config import *
from .physics import get_model_matrix, get_view_matrix, get_projection_matrix