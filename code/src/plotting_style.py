import matplotlib
import seaborn as sns

def setup():
    sns.set_context('paper', rc = {
        'font.family'    : 'serif',
        'font.serif'     : ['Times', 'Palatino', 'Computer Modern Roman'],
        'font.size'      : 10,
        'axes.titlesize' : 10,
        'axes.labelsize' : 10,
        'text.usetex'    : True,
        })
    matplotlib.rc('text', usetex=True)
