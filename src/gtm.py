import numpy as np
import bayespy as bp

from sklearn.preprocessing import normalize

# Fail fast on floating point error.
np.seterr(all="raise")
# But ignore underflow
np.seterr(under="ignore")

# TODO: Remove the observe step; we want to use SVI.
def setup_gtm(docs, K=10, T=100, alpha=1e-6, gamma=1e-6, a=1e-6, b=1e-6, initialize_from_prior=False):
    # Compute dimensions
    n_docs = len(docs)
    D = docs[0].shape[1]
    
    # Topic plate
    beta  = bp.nodes.Dirichlet(alpha * np.ones(T), plates=(K,), name="beta")
    
    # Clusters plate.
    # We implement diagonal Gaussians by using D independent Gaussians per plate.
    tau   = bp.nodes.Gamma(1e-6, 1e-6, plates=(D,T), name="tau")    
    mu    = bp.nodes.GaussianARD(0, tau, plates=(D,T), name="mu")

    # Local variables
    thetas = [None] * n_docs
    zs     = [None] * n_docs
    cs     = [None] * n_docs
    xs     = [None] * n_docs
    
    if initialize_from_prior:
        beta.initialize_from_prior()
        tau.initialize_from_prior()
        mu.initialize_from_prior()                
    
    for n in range(n_docs):
        n_words   = docs[n].shape[0]
        
        thetas[n] = bp.nodes.Dirichlet(gamma * np.ones(K), name="thetas[n]")
        
        # The following variables are plated according to n_words.        
        zs[n]     = bp.nodes.Categorical(thetas[n], plates=(n_words,1), name="zs[n]")
        
        # The cluster assignment is a mixture of categoricals.
        cs[n]     = bp.nodes.Mixture(zs[n], bp.nodes.Categorical, beta, name="cs[n]")
        
        # Finally, the output is a mixture of Gaussians
        xs[n]     = bp.nodes.Mixture(cs[n], bp.nodes.GaussianARD, mu, tau, name="xs[n]")
    
        # This is necessary to get a finite bound prior to any mean field iterations,
        # so we can run SVI. However, you must randomize the indicators prior to the
        # first mean field iterations, or you will get garbage results.
        
        if initialize_from_prior:        
            thetas[n].initialize_from_prior()                
            zs[n].initialize_from_prior()
            cs[n].initialize_from_prior()

        # vvv Evaluate the veracity of the statement below vvv
        # IMPORTANT! All the xs[n] must be observed from the get-go. Otherwise the
        # algorithm will attempt a full-covariance variational approximation.
        xs[n].observe(docs[n])        
    
    model_vars = [beta, tau, mu] + thetas + zs  + cs + xs
    
    finite_elbos = [np.isfinite(v.lower_bound_contribution()) for v in model_vars]
    assert np.all(finite_elbos), "Some variable was initialized to have infinite lower bound."
    
    if False:
        for i, v in enumerate(thetas):
            print("theta[%d] = %g" % (i, v.lower_bound_contribution()))
        for i, v in enumerate(zs):
            print("zs[%d] = %g" % (i, v.lower_bound_contribution()))
        for i, v in enumerate(cs):
            print("cs[%d] = %g" % (i, v.lower_bound_contribution()))
        
    vars_dict = { "beta": beta, 
                  "tau": tau,
                  "mu": mu,
                  "thetas": thetas,
                  "zs": zs,
                  "cs": cs,
                  "xs": xs }
    
    Q = bp.inference.VB(*model_vars)
    
    if initialize_from_prior:
        print("Initial ELBO, initialized from prior = %g" % Q.loglikelihood_lowerbound())
        
    return Q, vars_dict
    
def randomize_indicators(vars_dict, n):   
    vars_dict["zs"][n].initialize_from_random()
    vars_dict["cs"][n].initialize_from_random()    
    
def run_gtm_batch(Q, vars_dict, docs, max_iter=100):
    for n in range(len(docs)):
        randomize_indicators(vars_dict, n)
        
    Q.update(repeat=max_iter)

    # Conjugate gradients optimization FAIL. Just stick with MF instead.
#    Q.optimize(*Q.model, maxiter=max_iter, collapsed=vars_dict["zs"] + vars_dict["cs"])
#    Q.optimize(*Q.model, riemannian=True, maxiter=max_iter, collapsed=vars_dict["zs"])
    
    
# rm_params: Robbins Monro parameters. (init_step, forget, exponent)    
def run_gtm_svi(Q, vars_dict, docs, rm_params=(1e-2, 1, 0.7), n_epochs=100, n_minibatch=1, max_local_iters=10):
    Q.ignore_bound_checks = True
        
    n_docs = len(docs)
    
    mb_multiplier = n_docs / n_minibatch
    
    n_iters = int(np.ceil(n_epochs * mb_multiplier))
    
    global_vars = [vars_dict["beta"], vars_dict["tau"], vars_dict["mu"]]
        
    def local_vars_doc(n):
        return [vars_dict[name][n] for name in ["thetas", "zs", "cs", "xs"]]
    
    times_sampled = np.zeros(n_docs)
    elbos = np.zeros(n_iters)
    for t in range(n_iters):
        minibatch = np.random.choice(n_docs, size=n_minibatch, replace=False)
        new_docs  = [n for n in minibatch if times_sampled[n] == 0]
        
#        print("new_docs", new_docs)
        
        for n in new_docs:
            randomize_indicators(vars_dict, n)
        
        times_sampled[minibatch] += 1        
        
        local_vars = sum(map(local_vars_doc, minibatch), [])
        Q.update(*local_vars, repeat=max_local_iters, verbose=False)
                
        # SVI step
        step_sz = rm_params[0] * (t + rm_params[1]) ** (-rm_params[2])
        Q.gradient_step(*global_vars, scale=mb_multiplier * step_sz)
        
        # This will not be a good estimate...
        elbos[t] = Q.loglikelihood_lowerbound()
        
        print("SVI iter %d / %d (Epoch %d / %d), ELBO = %g" % (
                t, n_iters, np.floor(t / mb_multiplier), n_epochs, elbos[t]))

    return elbos        
        

#Q, vars_dict = setup_gtm(docs)
#run_gtm_batch(Q, vars_dict, docs)
#run_gtm_svi(Q, vars_dict, docs, n_minibatch=69)
#run_gtm_svi(Q, vars_dict, docs, n_minibatch=1)
#run_gtm_svi(Q, vars_dict, docs, n_minibatch=70)

# Eventually look at the recovered theta values, but for now, I have good enough assurance.

# In[23]:

def var_to_categorical(v):    
    return np.argmax(np.squeeze(v.get_moments()[0], 1), 1)

