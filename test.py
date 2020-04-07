import mbpoker
from evolve import Individual
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Syntax: python %s <generation>" % sys.argv[0])
        exit(1)

    gen = sys.argv[1]
    with open('data/generation_%s.pickle' % gen, 'rb') as f:
        pop = pickle.load(f)
    mbpoker.play_poker(pop[0].genome, pop[-1].genome, verbose=1)
