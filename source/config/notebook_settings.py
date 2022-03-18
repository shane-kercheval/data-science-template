# common imports
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math
import helpsk as hlp
import plotly.express as px
import plotly.io as pio

pio.renderers.default='notebook'

# pandas display options
# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#available-options
pd.options.display.max_columns = 30 # default 20
pd.options.display.max_rows = 60 # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200 # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# np.set_printoptions(edgeitems=3) # default 3

import matplotlib
from matplotlib import pyplot as plt

figure_size = (hlp.plot.STANDARD_WIDTH / 1.25, hlp.plot.STANDARD_HEIGHT / 1.25)

plot_params = {
    'figure.figsize': figure_size, 
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'xtick.labelsize': 'medium',
    'ytick.labelsize':'medium',
    'figure.dpi': 100,
}
# adjust matplotlib defaults
matplotlib.rcParams.update(plot_params)

import seaborn as sns
sns.set_style("darkgrid")
