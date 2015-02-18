from daft_init import *

G = da.PGM([6, 4], origin=[-.5, -1.5], node_unit=1.3, directed=True)

G.add_node(da.Node("alpha",   r"$\alpha$",   0, 1, fixed=True))

# # Document plate
G.add_plate(da.Plate([0.5, -0.45, 4, 1.95], label=r"$D$"))

G.add_node(da.Node("theta_d", r"$\theta_d$", 1, 1))

# ## Words subplate
G.add_plate(da.Plate([1.4, 0.55, 3, 0.9], label=r"$N_d$"))

G.add_node(da.Node("z_dn",    r"$z_{dn}$",  2, 1))
G.add_node(da.Node("w_dn",    r"$w_{dn}$",  4, 1, observed=True))

# ## Images subplate
G.add_plate(da.Plate([1.4, -0.40, 3, 0.8], label=r"$M_d$"))

G.add_node(da.Node("z_di",    r"$z_{di}$",  2, 0))
G.add_node(da.Node("v_di",    r"$v_{di}$",  3, 0))
G.add_node(da.Node("x_di",    r"$x_{di}$",  4, 0, observed=True))

# ## Word Vector Plates
G.add_plate(da.Plate([4.55, 0.55, 0.9, 0.9], label=r"$V$"))

G.add_node(da.Node("lambda_v",r"$\lambda_v, \sigma_{\lambda_v}$", 5, 1))

# ## Context Vector Plates
G.add_plate(da.Plate([4.55, 1.55, 0.9, 0.9], label=r"$B$"))
G.add_node(da.Node("pi_c",    r"$\pi_c, \sigma_{\pi_c}$",         5, 2))

# ## Topic Plates
G.add_plate(da.Plate([4.55, -0.45, 0.9, 0.9], label=r"$K$"))
G.add_node(da.Node("mu_k",    r"$\mu_k, \Sigma_k$",               5, 0))


# ## CNN Plates
G.add_plate(da.Plate([3.5, -1.45, 1.9, 0.9], label=r"$L$"))
G.add_node(da.Node("omega",   r"$\Omega, M$", 3, -1, fixed=True))
G.add_node(da.Node("y_l",     r"$y_{\ell}$",  4, -1, observed=True))
G.add_node(da.Node("x_l",     r"$x_{\ell}$",  5, -1, observed=True))

# Edges
G.add_edge("alpha", "theta_d")
G.add_edge("theta_d", "z_dn")
G.add_edge("theta_d", "z_di")
G.add_edge("z_dn", "w_dn")
G.add_edge("z_di", "v_di")
G.add_edge("x_di", "v_di")

G.add_edge("lambda_v", "w_dn")
G.add_edge("pi_c", "w_dn")

G.add_edge("mu_k", "w_dn")
G.add_edge("mu_k", "v_di")

G.add_edge("x_l", "y_l")
G.add_edge("omega", "v_di")
G.add_edge("omega", "y_l")

G.render()
G.figure.savefig("gtm_cnn.pdf")
