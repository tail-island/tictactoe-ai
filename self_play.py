import numpy as np
import pickle

from datetime     import datetime
from funcy        import *
from keras.models import load_model
from pathlib      import Path
from pv_mcts      import boltzman, pv_mcts_scores
from tictactoe    import State, popcount


MAX_GAME_COUNT      = 500  # AlphaZeroでは25000。
MCTS_EVALUATE_COUNT = 20   # AlphaZeroでは1600。
TEMPERATURE         = 1.0


def first_player_value(ended_state):
    if ended_state.lose:
        return 1 if (popcount(ended_state.pieces) + popcount(ended_state.enemy_pieces)) % 2 == 1 else -1

    return 0


def play(model):
    states = []
    ys = [[], None]

    state = State()

    while True:
        if state.end:
            break

        scores = pv_mcts_scores(model, MCTS_EVALUATE_COUNT, state)

        policies = [0] * 9
        for action, policy in zip(state.legal_actions, boltzman(scores, 1.0)):
            policies[action] = policy

        states.append(state)
        ys[0].append(policies)

        state = state.next(np.random.choice(state.legal_actions, p=boltzman(scores, TEMPERATURE)))

    value = first_player_value(state)
    ys[1] = tuple(take(len(ys[0]), cycle((value, -value))))

    return states, ys


def write_data(states, ys, game_count):
    y_policies, y_values = ys
    now = datetime.now()

    for i in range(len(states)):
        with open('./data/{:04}-{:02}-{:02}-{:02}-{:02}-{:02}-{:04}-{:02}.pickle'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, game_count, i), mode='wb') as f:
            pickle.dump((states[i], y_policies[i], y_values[i]), f)


def main():
    model = load_model(last(sorted(Path('./model').glob('*.h5'))))

    for i in range(MAX_GAME_COUNT):
        states, ys = play(model)

        print('*** game {:03}/{:03} ended at {} ***'.format(i + 1, MAX_GAME_COUNT, datetime.now()))
        print(states[-1])

        write_data(states, ys, i)


if __name__ == '__main__':
    main()
