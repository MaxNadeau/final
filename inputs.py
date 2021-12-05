import numpy as np

# prisoner's dilemma payoff matrix
pd_p_mat = np.array([[[4, 4], [0, 5]],
                     [[5, 0], [2, 2]]])

# chicken
ch_p_mat = np.array([[[0, 0], [0, 2]],
                     [[2, 0], [-4, -4]]])

# matching pennies
mp_p_mat = np.array([[[1, -1], [-1, 1]],
                     [[-1, 1], [1, -1]]])

# hawk-dove
hd_p_mat = np.array([[[-1, -1], [10, 0]],
                     [[0, 10], [5, 5]]])

# hawk-dove
hd_p_mat2 = np.array([[[-4, -4], [10, 0]],
                     [[0, 10], [5, 5]]])

# hawk-dove
hd_p_mat3 = np.array([[[-10, -10], [10, 0]],
                     [[0, 10], [5, 5]]])

# 2-action, uniform initial state population
unif_pop = []
for i in range(100):
    prob_vec = [i/100, 1 - i/100]
    unif_pop.append(prob_vec)
