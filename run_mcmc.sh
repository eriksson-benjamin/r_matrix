#!/bin/bash 
# @ input = /dev/null
# @ output = /home/beriksso/mcmc_ll_05.out
# @ error = /home/beriksso/mcmc_ll_05.err
# @ initialdir = /home/beriksso
# @ notify_user = beriksso
# @ notification = complete
# @ queue
cd /home/beriksso/TOFu/analysis/benjamin/other/tt_resonance/r_matrix
python mcmc_r_matrix.py
