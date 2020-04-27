from MarkovNetwork import MarkovNetwork
import numpy as np
import random
np.random.seed(1)

# A K Q game as described here: https://web.mit.edu/willma/www/2013lec3.pdf (slide 23)
# Colin will be an optimal agent, MNBs will be selected by GA to compete with Colin the optimal agent


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
            # compare cards
            rose_outcome, colin_outcome = compare_cards(rose_card, colin_card)
        # Colin folds and rose wins
        else:
            rose_outcome, colin_outcome = 1, 0
    # Rose checks
    else:
        # Showdown
        rose_outcome, colin_outcome = compare_cards(rose_card, colin_card)

    return rose_outcome, colin_outcome


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

if __name__ == '__main__':
    # 1-hot encoded cards (i.e. A is 1 0 0), (K is 0 1 0), (Q is 0 0 1)
    num_input_states = 3
    # Rose can either check=0 or bet=1
    num_output_states = 1

    rose_prime = gen_mn(num_input_states, num_output_states)

    rose_outcome, colin_outcome = play_game(rose_prime, None)
    print(rose_outcome, colin_outcome)