from MarkovNetwork import MarkovNetwork
import numpy as np

import pypokerengine.utils.visualize_utils as U
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

def get_last_action(round_state):
    street = round_state['street']
    hist = round_state['action_histories']
    if len(hist[street]) == 0:
        return None
    else:
        return hist[street][-1]

def to_binary_digits(n, n_bits):
    result = [0] * n_bits
    i = 0
    while n > 0 and i < n_bits:
        if n % 2 == 0:
            result[i] = 0
        else:
            result[i] = 1
        n //= 2
        i += 1
    return result

def to_number(bits):
    n = 0
    for i, bit in enumerate(bits):
        n *= 2
        if bit:
            n += 1
    return n

def card_str_to_numbers(card_str):
    suit = card_str[0]
    value = card_str[1]
    suitnums = {'C': 1, 'S': 2, 'H': 3, 'D': 4} # 0 is reserved for "no card"
    valuenums = {'A': 1, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
    suitnum = suitnums[suit]

    if value in '0123456789':
        valuenum = int(value)
    else:
        valuenum = valuenums[value]

    return to_binary_digits(suitnum, 3) + to_binary_digits(valuenum, 4)


class MarkovNetworkPlayer(BasePokerPlayer):
    def __init__(self, markov_network):
        self.markov_network = markov_network

    def declare_action(self, valid_actions, hole_card, round_state):
        hcs = [ card_str_to_numbers(hc) for hc in hole_card ]
        ccs = [ card_str_to_numbers(cc) for cc in round_state['community_card'] ]
        while len(ccs) < 5: # pad it to 5 community cards
            ccs.append( [0, 0, 0,  0, 0, 0, 0] )

        last_action = get_last_action(round_state)
        if last_action is None:
            last_action_type = 0
            last_action_amount = 0
        elif last_action['action'] == 'FOLD':
            last_action_type = 1
            last_action_amount = 0
        elif last_action['action'] == 'CALL':
            last_action_type = 2
            last_action_amount = last_action['amount']
        elif last_action['action'] == 'RAISE':
            last_action_type = 3
            last_action_amount = last_action['amount']
        else:
            last_action_type = 0
            last_action_amount = 0

        hand_strength_estimate = estimate_hole_card_win_rate(
            nb_simulation=100,
            nb_player=len(round_state['seats']),
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(round_state['community_card']))

        # 32 is arbitrary here
        hand_strength = int(hand_strength_estimate * (2**5))

        pot_size = round_state['pot']['main']['amount']

        input_vec = sum(hcs, []) + sum(ccs, []) \
            + to_binary_digits(last_action_type, 2) \
            + to_binary_digits(last_action_amount, 10) \
            + to_binary_digits(hand_strength, 5) \
            + to_binary_digits(pot_size, 10)

        self.markov_network.update_input_states(input_vec)
        self.markov_network.activate_network()

        output_states = self.markov_network.get_output_states()
        possible_actions = []

        output_bit_0 = output_states[0]
        output_bit_1 = output_states[1]
        output_raise_amount = to_number(output_states[2:4])

        if output_bit_0 and output_bit_1:
            return 'fold', 0
        elif output_bit_0:
            if output_raise_amount == 0:
                # min raise
                real_raise_amount = valid_actions[2]['amount']['min']
            elif output_raise_amount == 1:
                real_raise_amount = int(pot_size/4)
            elif output_raise_amount == 2:
                real_raise_amount = int(pot_size/2)
            elif output_raise_amount == 3:
                real_raise_amount = int(pot_size)

            return 'raise', real_raise_amount
        else:
            return 'call', valid_actions[1]['amount']

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def play_poker(genome_1, genome_2, verbose=0):
    net_1 = MarkovNetwork(num_input_states=76,
                          num_memory_states=30,
                          num_output_states=4,
                          genome=genome_1,
                          probabilistic=False)

    net_2 = MarkovNetwork(num_input_states=76,
                          num_memory_states=30,
                          num_output_states=4,
                          genome=genome_2,
                          probabilistic=False)

    game_config = setup_config(max_round=20,
                               initial_stack=100,
                               small_blind_amount=5)

    game_config.register_player(name='mb player 1', algorithm=MarkovNetworkPlayer(net_1))
    game_config.register_player(name='mb player 2', algorithm=MarkovNetworkPlayer(net_2))

    game_result = start_poker(game_config, verbose=verbose)
    return game_result
    

if __name__ == '__main__':
    genome_1 = np.random.randint(0, 256, 150000)
    genome_2 = np.random.randint(0, 256, 150000)
    print(play_poker(genome_1, genome_2))
