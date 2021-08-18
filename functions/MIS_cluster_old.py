"""Usage: example_run.py -S VALUE -W VALUE -o FILE 

-S VALUE        subject
-W VALUE        window
-o FILE         output file (npz)
-h --help       show this
"""

import numpy as np
import matplotlib.pyplot as plt  
from docopt import docopt
import time 
import pickle
import xarray            as xr
from   jpype             import * # pip install --user jpype1
from   joblib            import Parallel, delayed
print('GOT IMPORT1', flush=True)

import os
import sys  
sys.path.insert(0, '/home/giovanni.rabuffo/fufo/')
from src import analysis, simulation  # Import analysis for fcd and clustering
sys.path.insert(0, '/home/giovanni.rabuffo/fufo/notebooks/')
from functions import Time_delay_embedding as TDE


print('GOT IMPORT2', flush=True)


def save_obj(obj, name ):
    with open('/home/giovanni.rabuffo/fufo/notebooks/MIS/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('/home/giovanni.rabuffo/fufo/data/Pierpaolo-MEG/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

print('GOT FUNCTIONS', flush=True)



if __name__ == '__main__':
    args      = docopt(__doc__)
    S         = int(args["-S"])
    W    = int(args["-W"])
    out_path  = args["-o"]

    print('GOT INPUTS', flush=True)
    
    #sys.stdout = open(f'{out_path}.log', 'w')

    #print('GOT INPUTS2', flush=True)

    Trials=load_obj('Trials')
    sig=Trials['%d'%S][W]

    print('GOT IMPORT TRIALS', flush=True)

    esig=analysis.go_edge(sig)

    taus = np.arange(1,129,dtype=int)
    nedges=3003
    tau_max = 128 
    nbin=64

    t0      = time.time()
    EMIS=TDE.TauLag(esig, tau_max, nedges, 'MI', nbin)
    CPU_TIME    = time.time() - t0
    print(f'CPU TIME {CPU_TIME}', flush=True)

    np.savez(out_path, EMIS = EMIS)