import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
import inputs
from scipy import stats

# non-convergence at 5000, 100, 10

t_steps = 30
pop_size = 50  # 100 and 500 fights converges for MP?
fights = 1000

# def round_to_ints(orig):
#     rd = np.floor(orig)
#     delta = orig - rd
#     sum_delta = round(np.sum(delta))

#     incrementees = sorted(range(len(orig)), key = delta.__getitem__, reverse = True)[:sum_delta]

#     np.sum(delta)
#     rounded_down = {peer_id: int(bw_f) for peer_id, bw_f in peer_bws.items()}
#     delta  = {peer_id: peer_bws[peer_id] - rounded_down[peer_id] for peer_id in peer_bws.keys()}
#     sum_delta = round(sum(delta.values()))

#     assert  -0.000001 < sum_delta - sum(delta.values()) < 0.000001 and sum_delta < len(peer_bws)
#     incrementees = sorted(peer_bws.keys(), key = delta.__getitem__, reverse = True)[:sum_delta]
#     ret = {}
#     for peer_id in peer_bws.keys():
#         ret[peer_id] = rounded_down[peer_id] if peer_id not in incrementees else rounded_down[peer_id] + 1

#     print(f"Rounded version {ret}")
#     return ret


def fight(s0, s1, p_mat, sample=False):
    """
    s1, s2 are np prob-vectors, payoff matrix is m*m*2
    """
    # move0 = p_mat[np.random.choice(range(p_mat.shape[0]), p=s0)]
    # move1 = move0[np.random.choice(range(p_mat.shape[0]), p=s1)]
    # return tuple(move1)

    if sample:
        # random integer between 0 and 1
        signal = random.randint(0, 1)
        strat0 = s0[2*signal: 2*signal+2]
        strat1 = s1[2*(1-signal): 2*(1-signal)+2]
        move0 = p_mat[np.random.choice(range(p_mat.shape[0]), p=strat0)]
        move1 = move0[np.random.choice(range(p_mat.shape[0]), p=strat1)]
        return tuple(move1)
    else:
        ret = [0, 0]
        for signal in [0, 1]:
            strat0 = s0[2*signal: 2*signal+2]
            strat1 = s1[2*(1-signal): 2*(1-signal)+2]
            ret[0] += np.sum(np.outer(strat0, strat1) * p_mat[:, :, 0])
            ret[1] += np.sum(np.outer(strat0, strat1) * p_mat[:, :, 1])
        return ret[0]/2, ret[1]/2  # since 1/2 probability of each signal


def evolve(p, p_mat):

    def new_pop(f, p, fertile_prop=0.6, eps=0.01):
        # Note: Will not always produce exactly 100 agents
        n_reproducing = int(len(p) * fertile_prop)
        # gets indices of top n_r agents
        reproducers = np.argpartition(f, -n_reproducing)[-n_reproducing:]
        fitnesses = f[reproducers] - np.min(f[reproducers])
        # fitnesses = f - np.min(f)
        fitnesses /= np.max(fitnesses) + eps
        # fitnesses = fitnesses ** 1/3
        if np.min(fitnesses) < 0:
            print(f[reproducers], np.min(f[reproducers]), fitnesses)
        assert np.min(fitnesses) >= 0
        kids = np.rint(fitnesses / np.sum(fitnesses)
                       * pop_size)  # round to nearest int
        num_kids = np.sum(kids)
        print(f"NUMBER OF KIDS IS {num_kids}")
        print(f"KIDS FOR EACH PLAYER IS: {kids}")
        if np.min(kids) < 0 or np.min(kids.astype(int)) < 0:
            print(kids, fitnesses, fitnesses / np.sum(fitnesses) * pop_size)
        assert np.min(kids) >= 0
        #print(f"Reproducers are {np.repeat(p[reproducers], int(1/fertile_prop), axis=0)}")
        #print(kids, kids.dtype)
        print("Boutta spawn")
        # print(np.round(np.transpose(np.array([np.transpose(p[reproducers][:, 0]), f[reproducers], kids])), 3))

        # return np.repeat(p[reproducers], kids.astype(int), axis=0)
        return np.repeat(p[reproducers], kids.astype(int), axis=0)

    def add_noise(init_array):
        # Takes in a 2D numpy array, where each subarray is a strategy vector. Then adds noise to each subarray
        ret = init_array + \
            np.random.normal(loc=0, scale=0.01, size=init_array.shape)
        ret = np.abs(ret)
        ret = np.clip(ret, 0, 1)
        # new axis gets broadcasting to work
        ret[:, :2] = ret[:, :2] / np.sum(ret[:, :2], axis=1)[:, None]
        ret[:, -2:] = ret[:, -2:] / np.sum(ret[:, -2:], axis=1)[:, None]
        return ret

    f = np.zeros(p.shape[0])  # fitness of each individual

    avg_H_p = np.mean(p[:, 0])
    # print(avg_H_p)

    # for fighter in p:
    #     avg_H_p_opps = (avg_H_p - (fighter[0]/len(p))) * (len(p)/(len(p) - 1))
    #     #print(avg_H_p_opps)
    #     f0, _ = fight(fighter, np.array([avg_H_p_opps, 1 - avg_H_p_opps]), p_mat)
    #     fighter += f0

    for _ in range(fights):
        # sample two fighters randomly from the population
        fighters = np.random.choice(
            np.arange(len(p)), size=(2,), replace=False)
        f0, f1 = fight(p[fighters[0]], p[fighters[1]], p_mat)
        f[fighters[0]] += f0
        f[fighters[1]] += f1

    new_p = new_pop(f, p)
    new_p = add_noise(new_p)
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
    print(f"PVALUE IS {pval}")
    return pval


def main():
    # strats = (coop, defect)  # tuple of possible strategies for this game

    # 2-action, uniform initial state population
    unif_pop = []
    for i in range(pop_size):
        # create a random strategy vector, with probabilities summing to 1
        go = np.random.uniform(0, 1, size=2)
        ret = np.zeros(4)
        ret[::2] = go
        ret[1::2] = 1 - go
        unif_pop.append(ret)

    p = np.array(unif_pop)

    p_mat = inputs.hd_p_mat2
    #p = np.choice(strats, size=pop_size, replace=True, p=init_p)

    # OLD SETUP
    # mins = np.zeros(t_steps)
    # maxes = np.zeros(t_steps)
    # means = np.zeros(t_steps)
    # for t in range(t_steps):
    #     #print(f"Time={t}: strategies: {np.round(p, 3)}")
    #     p = evolve(p, p_mat)
    #     mins[t] = np.min(p[:,0])
    #     maxes[t] = np.max(p[:,0])
    #     means[t] = np.mean(p[:,0])
    # print(f"Final sums {np.sum(p, axis=0)}")

    # POTENTIAL NEW SETUP
    mins = np.array([])
    maxes = np.array([])
    means = np.array([])
    old_ps = p
    p = evolve(p, p_mat)
    t = 0
    while (test_convergence(p, old_ps) <= 0.995):
        print(f"Time={t}: strategies: {np.round(p, 3)}")
        t += 1
        old_ps = p
        p = evolve(p, p_mat)
        mins = np.append(mins, np.min(p[:, 0]))
        maxes = np.append(maxes, np.max(p[:, 0]))
        means = np.append(means, np.mean(p[:, 0]))
    print(f"Final sums {np.sum(p, axis=0)}")
    print(f"Total number of time steps was {t}")

    plt.plot(maxes, color="r", label="max p(H)")
    plt.plot(means, color="xkcd:orange", label="mean p(H)")
    plt.plot(mins, "y", label="min p(H)")
    #plt.axhline(y=5/6, color='r', linestyle='-')
    plt.axhline(y=sym_mat_msne(p_mat), color='b', linestyle='-')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plt.plot([1, 2, 3], color="y", marker="o")
    # plt.plot([1, 2, 3], color="r", marker="o")
    # plt.show()
    # print(plt.rcParams['backend'])
    main()
