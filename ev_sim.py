import numpy as np

t_steps = 10
fights = 100
pop_size = 100
init_p = np.array([0.5, 0.5])

# prisoner's dilemma payoff matrix
pd_p_mat = np.array([[[2, 2], [-3, 0]],
                     [[0, -3], [1, 1]]])
coop = np.array([1, 0])
defect = np.array([0, 1])

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


def fight(s0, s1, p_mat):
    """
    s1, s2 are np prob-vectors, payoff matrix is m*m*2
    """
    move0 = np.random.choice(p_mat, p=s0)
    move1 = np.random.choice(move0, p=s1)
    return tuple(move1)


def evolve(p, strats, p_mat):

    def new_pop(f, p, fertile_prop=0.5):
        # Note: Will not always produce exactly 100 agents
        n_reproducing = int(pop_size * fertile_prop)
        # gets indices of top n_r agents
        reproducers = np.argpartition(f, -n_reproducing)[-n_reproducing]
        return np.repeat(p[reproducers], int(1/fertile_prop))

    f = np.zero_like(p)  # fitness of each individual

    for _ in range(fights):
        # sample two fighters randomly from the population
        fighters = np.random.choice(
            np.arange(len(p)), size=(2,), replace=False)
        f0, f1 = fight(p[fighters[0]], p[fighters[1]], p_mat)
        f[fighters[0]] += f0
        f[fighters[1]] += f1

    new_p = new_pop(f, p)

    return new_pop(f, p)


def add_noise(init_array):
    # Takes in a 2D numpy array, where each subarray is a strategy vector. Then adds noise to each subarray
    ret = init_array + \
        np.random.normal(loc=0, scale=0.03, size=init_array.shape)
    ret = np.abs(ret)
    ret = np.clip(ret, 0, 1)
    ret /= np.sum(ret, axis=1)
    return ret


def main():
    strats = (coop, defect)  # tuple of possible strategies for this game

    p_mat = pd_p_mat
    p = np.choice(strats, size=pop_size, replace=True, p=init_p)

    for t in range(t_steps):
        print(f"Time={t}: strategy proportions {p}")
        p = evolve(p, strats, p_mat)


if __name__ == "__main__":
    main()
