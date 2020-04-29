from MarkovNetwork import MarkovNetwork
import numpy as np
import random
import matplotlib.pyplot as plt

np.random.seed(1)

# A K Q game as described here: https://web.mit.edu/willma/www/2013lec3.pdf (slide 23)
# Colin will be an optimal agent, MNBs will be selected by GA to compete with Colin the optimal agent


# Define some useful helper functions

# Generates a candidate MNB to play against Colin
def gen_mn(num_in, num_out):
    return MarkovNetwork(num_input_states=num_in,
                          num_memory_states=1,
                          num_output_states=num_out,
                          seed_num_markov_gates=3,
                          probabilistic=True)

def play_game(rose, colin):

    deck = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    random.shuffle(deck)
    rose_card, colin_card = deck[:2]

    # Colins action depends on Rose, because if Rose checks Colin always checks thus only get Colin's action if Rose bets

    rose.update_input_states(rose_card)
    rose.activate_network()
    rose_action = rose.get_output_states()
    rose.states[-1] = 0

    # todo: check if rose_action list or not
    if rose_action == 1:
        colin_action = get_colin_action(colin_card)

        if colin_action == 1:
            # Colin calls compare cards
            rose_outcome, colin_outcome = compare_cards(rose_card, colin_card)

            if rose_outcome:
                rose_winnings, colin_winnings = 2, -2
            else:
                rose_winnings, colin_winnings = -2, 2

        # Colin folds and rose wins
        else:
            rose_winnings, colin_winnings = 1, -1

    # Rose checks
    else:
        # Showdown
        rose_outcome, colin_outcome = compare_cards(rose_card, colin_card)

        if rose_outcome:
            rose_winnings, colin_winnings = 1, -1
        else:
            rose_winnings, colin_winnings = -1, 1

    return rose_winnings, colin_winnings


# No ties in A K Q game
def compare_cards(card1, card2):
    if np.nonzero(card1)[0][0] < np.nonzero(card2)[0][0]:
        # First card wins return 1, 0
        return 1, 0
    else:
        return 0, 1


def get_colin_action(card):
    # Always call an Ace
    if card[0] == 1:
        return 1
    elif card[1] == 1:
        roll = np.random.uniform()
        return 1 if roll < .33 else 0
    # Always folds a Queen
    else:
        return 0


def battle(rose_candidate):
    # Let rose_prime battle Colin over 100 games and see who wins!
    rose_total = 0
    colin_total = 0
    for _ in range(100):
        rose_winnings, colin_winnings = play_game(rose_candidate, None)
        rose_total += rose_winnings
        colin_total += colin_winnings

    return rose_total, colin_total


def init_population(M):
    return [gen_mn(3,1) for _ in range(M)]


def mutate(parents, p_m=0.1, p_pm=0.1, p_cm=0.001):
    children = np.empty(parents.shape[0], dtype=object)
    for i, parent in enumerate(parents):
        seed_genome = parent.genome
        gate_io_ranges, gate_pt_ranges = get_gate_ranges(seed_genome)
        if len(gate_pt_ranges) == 0:
            gate_pt_locs = np.array([], dtype=int)
        else:
            gate_pt_locs = np.hstack([ list(range(*r)) for r in gate_pt_ranges ])
        if len(gate_io_ranges) == 0:
            gate_io_locs = np.array([], dtype=int)
        else:
            gate_io_locs = np.hstack([ list(range(*r)) for r in gate_io_ranges ])

        # Modify the seed genome with some probability in each loci
        idx = np.random.choice([False, True], size=seed_genome.shape[0], p=[1 - p_m, p_m])

        # Only take positions from gate_pt_locs
        pt_idxs = np.random.choice([False, True], size=gate_pt_locs.shape[0], p=[1 - p_pm, p_pm])
        # Only take positions from gate_io_locs
        io_idxs = np.random.choice([False, True], size=gate_io_locs.shape[0], p=[1 - p_cm, p_cm])

        pt_idx = gate_pt_locs[pt_idxs]
        io_idx = gate_io_locs[io_idxs]

        seed_genome[idx] = np.random.randint(256, size=seed_genome[idx].shape)
        seed_genome[pt_idx] = np.random.randint(256, size=pt_idx.shape)
        seed_genome[io_idx] = np.random.randint(256, size=io_idx.shape)
        mn = MarkovNetwork(num_input_states=parent.num_input_states,
                           num_output_states=parent.num_output_states,
                           num_memory_states=parent.num_memory_states,
                           probabilistic=True,
                           genome=seed_genome)
        children[i] = mn
    return children

def get_gate_ranges(genome):
    ios = [] # input/output wirings
    pts = [] # probability tables
    for i in range(len(genome) - 1):
        if genome[i] == 42 and genome[i+1] == 213:
            i2 = i + 2
            num_inputs = (genome[i2] % MarkovNetwork.max_markov_gate_inputs) + 1
            i2 += 1
            num_outputs = (genome[i2] % MarkovNetwork.max_markov_gate_outputs) + 1
            i2 += 1
            io_range_size = MarkovNetwork.max_markov_gate_inputs \
                + MarkovNetwork.max_markov_gate_outputs
            ios.append( (i2, i2 + io_range_size) )
            i2 += io_range_size
            pt_range_size = (2**num_inputs) * (2**num_outputs)
            pts.append( (i2, i2 + pt_range_size) )

    return ios, pts

def visualize(trace, filename):
    plt.plot(np.arange(100), np.array(trace))
    plt.legend(('best', 'worst', 'average'))
    plt.title("Fitness in successive generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()
    plt.savefig(filename)

if __name__ == '__main__':
    M = 100  # Initial Pop. size
    N = 100  # Iterations of GA
    e = 4  # Elites
    p_m = .01  # Mutation rate

    R = M - e  # Parents selected by fitness proportionate selection, then mutation will occur

    # Generate initial population
    genotypes = np.array(init_population(M))
    trace = list()
    for _ in range(N):

        # Evaluate fitness of each genotype by battling against an optimal agent
        fitness = np.zeros(100)
        for idx, genotype in enumerate(genotypes):
            fitness[idx] = battle(genotype)[0]

        trace.append([max(fitness), min(fitness), np.mean(fitness)])
        print(max(fitness))

        # Select elites
        elite_idx = np.argsort(fitness)[::-1][:e]

        # Fill remaining population (R=M-e) with parents, who will be mutated
        pos_fitness = fitness - min(fitness)
        parents = genotypes[np.random.choice(M, size=R, p=(pos_fitness / sum(pos_fitness)))]

        # Mutate the parents
        children = mutate(parents, p_m=.1)
        genotypes = np.hstack((genotypes[elite_idx], children))

    visualize(trace, "mnb_run.png")

