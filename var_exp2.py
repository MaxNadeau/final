import numpy as np
import matplotlib.pyplot as plt
import inputs
from scipy import stats

t_steps = 150
pop_size = 50
fights = 1000

def fight(s0, s1, p_mat):
    """
    s1, s2 are np prob-vectors, payoff matrix is m*m*2
    """
    # this is EXPECTED payoff
    return np.sum(np.outer(s0, s1) * p_mat[:, :, 0]), np.sum(np.outer(s0, s1) * p_mat[:, :, 1])
    # this is actual payoff
    # move0 = p_mat[np.random.choice(range(p_mat.shape[0]), p=s0)]
    # move1 = move0[np.random.choice(range(p_mat.shape[0]), p=s1)]
    # return tuple(move1)


def evolve(p, p_mat):

    def new_pop(f, p, fertile_prop=1, eps=0.01):
        # Note: Will not always produce exactly pop_size agents
        n_reproducing = int(len(p) * fertile_prop)
        # gets indices of top n_r agents
        reproducers = np.argpartition(f, -n_reproducing)[-n_reproducing:]
        
        # subtracts min ONLY OF REPRODUCERS
        # fitnesses = f[reproducers] - np.min(f[reproducers])
        # subtracts min of WHOLE POPULATION
        fitnesses = f[reproducers] - np.min(f)

        fitnesses /= np.max(fitnesses) + eps
        assert np.min(fitnesses) >= 0
        kids = np.rint(fitnesses / np.sum(fitnesses)
                       * pop_size)  # round to nearest int
        # num_kids = np.sum(kids)
        # print(f"NUMBER OF KIDS IS {num_kids}")
        # print(f"KIDS FOR EACH PLAYER IS: {kids}")
        if np.min(kids) < 0 or np.min(kids.astype(int)) < 0:
            print(kids, fitnesses, fitnesses / np.sum(fitnesses) * pop_size)
        assert np.min(kids) >= 0
        #print(f"Reproducers are {np.repeat(p[reproducers], int(1/fertile_prop), axis=0)}")
        #print(kids, kids.dtype)
        # print("Boutta spawn")
        # print(np.round(np.transpose(np.array([np.transpose(p[reproducers][:, 0]), f[reproducers], kids])), 3))

        # return np.repeat(p[reproducers], kids.astype(int), axis=0)
        return np.repeat(p[reproducers], kids.astype(int), axis=0)

    def add_noise(init_array, sd_noise):
        # Takes in a 2D numpy array, where each subarray is a strategy vector. Then adds noise to each subarray
        ret = init_array + \
            np.random.normal(loc=0, scale=sd_noise, size=init_array.shape)
        ret = np.abs(ret)
        ret = np.clip(ret, 0, 1)
        # new axis gets broadcasting to work
        ret = ret / np.sum(ret, axis=1)[:, None]
        return ret

    f = np.zeros(p.shape[0])  # fitness of each individual

    for _ in range(fights):
        # sample two fighters randomly from the population
        fighters = np.random.choice(
            np.arange(len(p)), size=(2,), replace=False)
        f0, f1 = fight(p[fighters[0]], p[fighters[1]], p_mat)
        f[fighters[0]] += f0
        f[fighters[1]] += f1

    new_p = new_pop(f, p)
    new_p = add_noise(new_p, 0.01)
    return new_p


def sym_mat_msne(mat):
    a, b, c, d = mat[0, 0, 0], mat[0, 1, 0], mat[1, 0, 0], mat[1, 1, 0]
    return (b - d) / (b - a + c - d)

# Currently only works for two-action games, uses 2-sample Kolmogorov-Smirnov Test
# Looked at KL Divergence, maybe an alternative for other games
# Difficulty with this is that the null hypothesis is that the samples come from
# the same distribution... not ideal, may have to change. Couldn't find anything
# comparable where the null hypothesis was that they're different.
def test_convergence(new_ps, old_ps):
    # Converts vector format to single float format
    new_ps = new_ps[:, 0]
    old_ps = old_ps[:, 0]
    pval = stats.kstest(new_ps, old_ps).pvalue
    # print(f"PVALUE IS {pval}")
    return pval

def simulate(t_steps, p_mat, p):
    # mins = np.zeros(t_steps)
    # maxes = np.zeros(t_steps)
    means = np.zeros(t_steps)
    # twenty_fifths = np.zeros(t_steps)
    # seventy_fifths = np.zeros(t_steps)
    for t in range(t_steps):
        # mins[t] = np.min(p[:,0])
        # twenty_fifths[t] = np.quantile(p[:, 0], q = 0.25)
        # maxes[t] = np.max(p[:,0])
        means[t] = np.mean(p[:,0])
        # seventy_fifths[t] = np.quantile(p[:, 0], q = 0.75)
        # maxes[t] = np.max(p[:,0])
        p = evolve(p, p_mat)

    return means
    # print(f"Final sums {np.sum(p, axis=0)}")

    # plt.plot(maxes, color="r", label="max p(H)")
    # plt.plot(means, color="xkcd:orange", label="mean p(H)")
    # plt.axhline(y=sym_mat_msne(p_mat), color='b', linestyle='-')
    # plt.legend()
    # plt.show()
    
def main():
    # epsilon = 0.03
    p_mat = inputs.hd_p_mat3

    # half of population plays MSNE, other half plays MSNE + epsilon
    msne = sym_mat_msne(p_mat)
    # elt_msne = np.array([[msne, 1-msne]])
    # msne_vec = np.repeat(elt_msne, pop_size/2, axis=0)
    # msne_plus = msne + epsilon
    # elt_msne_plus = np.array([[msne_plus, 1-msne_plus]])
    # msne_plus_vec = np.repeat(elt_msne_plus, pop_size/2, axis=0)
    # p = np.concatenate((msne_vec, msne_plus_vec), axis=0)

    # uniform initial state population
    unif_pop = []
    for i in range(pop_size):
        prob_vec = [i/pop_size, 1 - i/pop_size]
        unif_pop.append(prob_vec)
    p = np.array(unif_pop)

    num_simulations = 50
    means = np.zeros_like(t_steps)
    for i in range(num_simulations):
        means_i = simulate(t_steps, p_mat, p)
        means = np.add(means, means_i)
    means = means / num_simulations
    plt.plot(means, color="xkcd:orange", label="mean p(H)")
    plt.axhline(y=msne, color='b', linestyle='-', label="MSNE")
    # plt.axhline(y=msne + epsilon, color='r', linestyle='-', label="MSNE + epsilon")
    # plt.axhline(y=msne + epsilon/2, color='g', linestyle='-', label="MSNE + half epsilon")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plt.plot([1, 2, 3], color="y", marker="o")
    # plt.plot([1, 2, 3], color="r", marker="o")
    # plt.show()
    # print(plt.rcParams['backend'])
    main()