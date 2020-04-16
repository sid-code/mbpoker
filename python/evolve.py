import mbpoker
import pickle
import multiprocessing
import numpy as np
import uuid # for naming individuals
from log import logger
from glicko2 import glicko2

class Individual(glicko2.Player):
    def __init__(self, genome, generation=0):
        glicko2.Player.__init__(self)

        self.genome = genome
        self.network = mbpoker.make_markov_network(genome)
        self.name = uuid.uuid4()
        self.generation = generation
        self.glicko2 = 0

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

def make_initial_population(size, genome_size, seed_gates):
    result = []
    for _ in range(size):
        # we use this function from mbpoker to make a random markov
        # network because it's guaranteed to have gates
        genome = mbpoker.make_seed_genome(genome_size, seed_gates=seed_gates)
        result.append(Individual(genome))
    return result

def play_game(matchup, verbose=0):
    """Return the best individual in `matchup` (list of individuals).

    For now, there can only be 2 individuals in a matchup.

    The result will be determined by playing `n_games` games of poker
    and choosing the player who wins more games.

    """

    # The reason matchup is passed in as a tuple is to make this
    # multiprocessing-friendly.

    assert len(matchup) == 2, 'I can only compare two individuals'
    individual_1, individual_2 = matchup

    result = mbpoker.play_poker(individual_1.network, individual_2.network)
    better_player = max(result['players'], key=lambda x: x['stack'])['name']
    return better_player

def get_next_generation(population, game_count, parent_count,
                        mutation_count, crossover_count, verbose=0):
    pop_size = len(population)

    pairings = []

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in range(game_count):
        i1 = np.random.choice(population)
        i2 = np.random.choice(population)
        pairings.append( (i1, i2) )

    logger.info("Pairings determined, playing games...")
    results = pool.map(play_game, pairings)

    logger.info("Updating ratings...")
    for better_player, (p1, p2) in zip(results, pairings):
        if better_player == 1:
            winner, loser = p1, p2
        else:
            loser, winner = p1, p2

        winner_old_rating = winner.getRating()
        loser_old_rating = loser.getRating()

        winner_old_rd = winner.getRd()
        loser_old_rd = loser.getRd()

        winner.update_player([ loser_old_rating ], [ loser_old_rd ], [ 1 ])
        loser.update_player([ winner_old_rating ], [ winner_old_rd ], [ 0 ])

        winner_new_rating = winner.getRating()
        loser_new_rating = loser.getRating()

        winner_new_rd = winner.getRd()
        loser_new_rd = loser.getRd()

        if verbose:
            logger.debug('winner glicko2 %d +/- %d -> %d +/- %d' % (winner_old_rating, winner_old_rd,
                                                             winner_new_rating, winner_new_rd))
            logger.debug('loser glicko2 %d +/- %d -> %d +/- %d' % (loser_old_rating, loser_old_rd,
                                                            loser_new_rating, loser_new_rd))
    # The initial next generation is this generation sorted by rating descending
    next_generation = sorted(population, key=lambda x: -x.getRating())
    logger.info('highest 5 glicko2s: %s' % [ int(x.getRating()) for x in next_generation[:5] ])

    # Select parents
    parents = next_generation[:parent_count]

    # Now we have our list of winners who will get to reproduce. The
    # new individuals will be pushed to the front of the population
    # and the population will be trimmed to match its original size

    for _ in range(mutation_count):
        p = np.random.choice(parents)
        child = p.dup().mutate_point(count=10)
        next_generation.insert(0, child)

    for _ in range(crossover_count):
        # choose parents from winners
        p1 = np.random.choice(parents)
        p2 = np.random.choice(parents)
        child = p1.dup().cross_with(p2)
        next_generation.insert(0, child)

    return next_generation[:pop_size]

if __name__ == '__main__':
    gen_count = 50
    logger.info("Generating initial population")
    pop = make_initial_population(size=300, genome_size=30000, seed_gates=300)


    # Run through some generations
    for i in range(gen_count):
        avg_generation = sum(x.generation for x in pop) / len(pop)
        logger.info('generation %d' % (i))
        pop = get_next_generation(pop,
                                  game_count=10000,
                                  parent_count=30,
                                  mutation_count=100,
                                  crossover_count=100,
                                  verbose=1)

        with open('data/generation_%d.pickle' % i, 'wb') as f:
            pickle.dump(pop, f)

    # Now let's simulate a game between the first two individuals
    play_poker(pop[0].genome, pop[1].genome, verbose=1)
