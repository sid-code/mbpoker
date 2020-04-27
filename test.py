import mbpoker
from evolve import Individual, play_game
import random
import pickle
import sys

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Syntax: python %s <generation>" % sys.argv[0])
        exit(1)

    gen = sys.argv[1]
    with open('data/generation_%s.pickle' % gen, 'rb') as f:
        pop = pickle.load(f)
    pop.sort(key=lambda x: -x.getRating())
    # pick the two highest rated individuals
    p0, p1 = pop[0], pop[1]

    print(p0.getRating(), p1.getRating())

    # reset their ratings for fun
    p0.setRating(1500)
    p0.setRd(150)
    p1.setRating(1500)
    p1.setRd(150)


    for _ in range(1000):
        print("---ratings---")
        print("p0", p0.getRating(), p0.getRd())
        print("p1", p1.getRating(), p1.getRd())

        result = play_game([p0, p1], verbose=1)
        if result == 0:
            continue
        elif result == 1:
            winner = p0
            loser = p1
        elif result == 2:
            winner = p1
            loser = p0
        winner.record_win_against(loser)
            
