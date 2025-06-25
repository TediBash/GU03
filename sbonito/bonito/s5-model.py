"""Implementation of the Bonito-CTC model

Based on: 
https://github.com/nanoporetech/bonito
"""

import os
import sys
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from classes import BaseModelImpl
from s5 import S5Block


