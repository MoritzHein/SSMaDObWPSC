# Matplotlib configuration for the paper figures: sets figure sizes matching
# the IEEE Access column/text width, uses LaTeX text rendering and loads the
# same math notation (notation.tex) as the paper so that axis labels like
# $M$, $B$ match the manuscript. Imported by the visualization_* scripts.
import matplotlib as mpl

# In Latex, determine text and column width with:
#
# \usepackage{layouts}
# ...
# The column width is: \printinunitsof{in}\prntlen{\columnwidth}. 
# The text width is: \printinunitsof{in}\prntlen{\textwidth} 

textwidth = 7.08661 # in inches
columnwidth = 3.44488 # in inches
golden_ratio = 1.618
height = columnwidth/golden_ratio

fontsize = 10

mpl.rcParams.update({
    'lines.linewidth':1,
    'font.size': fontsize,
    'axes.titlesize'  : fontsize,
    'axes.labelsize'  : fontsize,
    'xtick.labelsize' : fontsize, 
    'ytick.labelsize' : fontsize,
    'font.serif': [],
    'axes.titlepad':  10,
    'axes.labelsize': 'medium',
    'figure.figsize': (columnwidth, height),
    'axes.grid': False,
    'lines.markersize': 5,
    'text.usetex': True,
    'pgf.rcfonts': False,    # don't setup fonts from rc parameters
    'axes.unicode_minus': False,
    "text.usetex": True,     # use inline math for ticks
    "pgf.texsystem" : "pdflatex",
    "pgf.rcfonts": False,    # don't setup fonts from rc parameters
})


# Load notation from file. The same notation is used in the paper.
with open('notation.tex', 'r') as f:
    tex_preamble = f.readlines()

tex_preamble = ''.join(tex_preamble)

mpl.rcParams.update({
    'text.latex.preamble': tex_preamble,
    'pgf.preamble': tex_preamble,
})

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
boxprops = dict(boxstyle='round', facecolor='white', alpha=0.5, pad = .4)