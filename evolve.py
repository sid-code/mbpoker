from mbpoker import play_poker, make_seed_genome
import pickle
import multiprocessing
import numpy as np
import uuid # for naming individuals

class Individual:
    def __init__(self, genome, generation=0):
        self.genome = genome
        self.name = uuid.uuid4()
        self.generation = generation

    def __str__(self):
        return self.name.hex[:12]

    def dup(self):
        # the duplicate has generation zero
        return Individual(self.genome[:], self.generation + 1)

    def mutate_point(self, count=1):
        # replace a random number in this genome with another random
        # number
        for _ in range(count):
            pos = np.random.randint(0, len(self.genome)-1)
            self.genome[pos] = np.random.randint(0, 255)
        return self
        
    def cross_with(self, other_individual):
        # choose a point in the genome and replace everything after it
        # with the other individual's genome
        crossover_point = np.random.randint(0, len(self.genome)-1)
        self.genome[crossover_point:] = other_individual.genome[crossover_point:]
        return self

def make_initial_population(size=100, genome_size=15000):
    result = []
    for _ in range(size):
        # we use this function from mbpoker to make a random markov
        # network because it's guaranteed to have gates
        genome = make_seed_genome()
        result.append(Individual(genome))
    return result

def determine_better_individual(matchup, n_games=3, verbose=0):
    """Return the best individual in `matchup` (list of individuals).

    For now, there can only be 2 individuals in a matchup.

    The result will be determined by playing `n_games` games of poker
    and choosing the player who wins more games.

    """

    # The reason matchup is passed in as a tuple is to make this
    # multiprocessing-friendly.

    assert len(matchup) == 2, 'I can only compare two individuals'
    individual_1, individual_2 = matchup

    wins_1 = 0
    wins_2 = 0
    for _ in range(n_games):
        result = play_poker(individual_1.genome, individual_2.genome)
        better_player = max(result['players'], key=lambda x: x['stack'])['name']
        if better_player == 1:
            wins_1 += 1
        else:
            wins_2 += 1

    if wins_1 > wins_2:
        winner = individual_1
    else:
        winner = individual_2

    if verbose:
        print('%s vs %s -> win: %s' % (individual_1, individual_2, winner))
    return winner

def get_next_generation(population, mutation_count, crossover_count):
    pop_size = len(population)

    next_generation = population[:]

    pairings = []

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in range(pop_size):
        i1 = np.random.choice(population)
        i2 = np.random.choice(population)
        pairings.append( (i1, i2) )

    winners = pool.map(determine_better_individual, pairings)

    # Now we have our list of winners who will get to reproduce. The
    # new individuals will be pushed to the front of the population
    # and the population will be trimmed to match its original size

    for _ in range(mutation_count):
        p = np.random.choice(winners)
        child = p.dup().mutate_point(count=10)
        next_generation.insert(0, child)

    for _ in range(crossover_count):
        # choose parents from winners
        p1 = np.random.choice(winners)
        p2 = np.random.choice(winners)
        child = p1.dup().cross_with(p2)
        next_generation.insert(0, child)

    return next_generation[:pop_size]

if __name__ == '__main__':
    gen_count = 50
    pop = make_initial_population(size=100, genome_size=10000)

    # Run through some generations
    for i in range(gen_count):
        avg_generation = sum(x.generation for x in pop) / len(pop)
        print('generation %d' % (i))
        pop = get_next_generation(pop, mutation_count=30,
                                  crossover_count=30)

        with open('data/generation_%d.pickle' % i, 'wb') as f:
            pickle.dump(pop, f)

    # Now let's simulate a game between the first two individuals
    play_poker(pop[0].genome, pop[1].genome, verbose=1)
