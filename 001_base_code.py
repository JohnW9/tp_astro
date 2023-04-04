#%% Snippet for left PC (Ctrl + Enter to run cell)
import os
os.chdir('c:\\tp_astro')

# %% Packages needed 
import tp_astro as tp
from miscmath import fit_circle

# %% Initialize communication (run cell once is enough)
cam, pos = tp.tp_init()

#%% Move around positioner
pos.goto_absolute(0,0)