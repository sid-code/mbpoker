from mbpoker import play_poker
import numpy as np
import uuid # for naming individuals

class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.name = uuid.uuid4()
    def __str__(self):
        return self.name.hex[:12]

    def dup(self):
        return Individual(self.genome[:])

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
    for __ in range(size):
        result.append(Individual(np.random.randint(0, 255, genome_size)))
    return result

def determine_better_individual(individual_1, individual_2, n_games=3):
    """Return the better of `individual_1` and `individual_2`.

    The result will be determined by playing `n_games` games of poker
    and choosing the player who wins more games.

    """
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
        return individual_1
    else:
        return individual_2

def get_next_generation(population):
    winners = []
    pop_size = len(population)
    next_generation = population[:]

    mutation_count = 5
    crossover_count = 5

    for _ in range(pop_size):
        i1 = np.random.choice(population)
        i2 = np.random.choice(population)
        winner = determine_better_individual(i1, i2)
        print("Pitting %s against %s, winner %s" % (i1, i2, winner))
        winners.append(winner)

    # Now we have our list of winners who will get to reproduce. The
    # new individuals will be pushed to the front of the population
    # and the population will be trimmed to match its original size

    for _ in range(mutation_count):
        p = np.random.choice(winners)
        child = winner.dup().mutate_point(count=10)
        next_generation.insert(0, child)

    for _ in range(crossover_count):
        # choose parents from winners
        p1 = np.random.choice(winners)
        p2 = np.random.choice(winners)
        child = p1.dup().cross_with(p2)
        next_generation.insert(0, child)

    return next_generation[:pop_size-1]

if __name__ == '__main__':
    gen_count = 10
    pop = make_initial_population(size=10, genome_size=10000)

    # Run through some generations
    for i in range(gen_count):
        print('generation %d: %s' % (i, ', '.join([ str(ind) for ind in pop])))
        pop = get_next_generation(pop)

    # Now let's simulate a game between the first two individuals
    play_poker(pop[0].genome, pop[1].genome, verbose=1)
