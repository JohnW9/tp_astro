# %% Packages needed (Ctrl + Enter to run cell)
import tp_astro as tp
from miscmath import fit_circle

# %% Initialize communication (run cell once is enough)
cam, pos = tp.tp_init()

#%% Move around positioner
pos.goto_absolute(0,0)