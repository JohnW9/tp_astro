# %% Packages needed
""" To run a cell: Ctrl+Enter """
import tp_astro as tp
from miscmath import fit_circle
import numpy as np
import matplotlib.pyplot as plt

# %% Initialize communication (run cell once is enough)
cam, pos = tp.tp_init()

#%% Move around positioner
pos.goto_absolute(0,0)
