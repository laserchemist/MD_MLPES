from .base import Frame, Curve1D, eigvals_sorted, slater_overlap, switch
from .london import LondonFrame
from .channel import ChannelFrame, Diabat

# DIMWaterFrame builds its O(2p^4) operator algebra on import (about a second), so it is imported on demand:
#   from diabml.frames.dim_oh import DIMWaterFrame
