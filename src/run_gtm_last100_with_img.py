# TODO: docopt
import numpy as np

import save_py3k as save
import gtm

# SILLY CONFIGURATION STUFF
# (eventually read from docopt)
K = 10
T = 100

docs = save.load2("../../synsetw2v/corpus2vec_nostem.pkl")[-100:]

total_vecs = sum(len(d) for d in docs)

print("Loaded %d docs and %d vectors." % (len(docs), total_vecs))

Q, vars_dict = gtm.setup_gtm(docs, K=K, T=T)
gtm.run_gtm_batch(Q, vars_dict, docs)

save.save("Q_res_last100.pkl", Q=Q, vars_dict=vars_dict)

