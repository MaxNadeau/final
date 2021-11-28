import numpy as np

t_steps = 10
fights = 100
init_p = np.array([0.5, 0.5])

pd_payoff = np.array([[[2, 2], [-3, 0]]
                       []])

def fight(s0, s1, payoff):
    """
    s1, s2 are np prob-vectors, payoff matrix is m*m*2
    """
    move0 = np.random.choice(payoff, p = s0)
    move1 = np.random.choice(move1, p = s1)
    return tuple(move1)

def evolve(p, strats, payoff):

    def fitnesses_to_proportions(f):
        return f / np.sum(f)

    f = np.zero_like(p)

    for _ in range(fights):

        fighters = np.random.choice(np.arange(len(strats)), size = (2,), p = p) # sample two fighters randomly from the population
        f0, f1 = fight(strats[fighters[0]], strats[fighters[1]], payoff)
        f[fighters[0]] += f0
        f[fighters[1]] += f1


    return fitnesses_to_proportions(f)



def main():
    strats = () # tuple of possible strategies for this game


    p = init_p
    for t in range(time_steps:



if __name__ = "__main__":
    main()
