from typing import List
import pandas as pd
from xmot.digraph.particle import Particle

"""
    Collection of constants and functions needed in all classes and modules.
"""

TIMEFRAMELENGHT = 1 / 45000 * 1000 # 45,000 frames per second, speed-up by x1000.
PIC_DIMENSION = [0, 0]             # Dimensions of picture in pixels.