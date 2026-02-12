import mlxp

import matplotlib.pyplot as plt
import numpy as np
import os

from imgcat import imgcat
import io

from viz import *



# Your plot appears as a real image inside the terminal!



# 1. Initialize the reader pointing to your output directory
# Note: mlxp usually expects the path to the directory containing 
# the individual experiment run folders.
path = "data/outputs/deterministic_9"
query = "info.status=='COMPLETE'"
results = mlxp.Reader(path).filter(query_string=query)
figures_dir = '../figures'


#plot_phase_portrait(results, figures_dir =figures_dir)

plot_spectral_stack(results, figures_dir =figures_dir)


plot_grad_norm(results, figures_dir =figures_dir)