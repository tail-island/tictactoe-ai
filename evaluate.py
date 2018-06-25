import numpy as np

from datetime     import datetime
from funcy        import *
from keras.models import load_model
from pathlib      import Path
from pv_mcts      import boltzman, pv_mcts_scores
from shutil       import copy
from tictactoe    import State, popcount


MAX_GAME_COUNT      = 100  # AlphaZeroでは400。
MCTS_EVALUATE_COUNT = 20   # AlphaZeroでは1600。
TEMPERATURE         = 1.0


def first_player_point(ended_state):
    if ended_state.lose:
        return 1 if (popcount(ended_state.pieces) + popcount(ended_state.enemy_pieces)) % 2 == 1 else 0

    return 0.5


def play(models):
    state = State()

    for model in cycle(models):
        if state.end:
            break;

        state = state.next(np.random.choice(state.legal_actions, p=boltzman(pv_mcts_scores(model, MCTS_EVALUATE_COUNT, state), TEMPERATURE)))

    return first_player_point(state)


def update_model():
    challenger_path = last(sorted(Path('./model/candidate').glob('*.h5')))
    champion_path   = last(sorted(Path('./model').glob('*.h5')))

    copy(str(challenger_path), str(champion_path.with_name(challenger_path.name)))


def main():
    models = tuple(map(lambda path: load_model(last(sorted(path.glob('*.h5')))), (Path('./model/candidate'), Path('./model'))))
    total_point = 0

    for i in range(MAX_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(models)
        else:
            total_point += 1 - play(tuple(reversed(models)))

        print('*** game {:03}/{:03} ended at {} ***'.format(i + 1, MAX_GAME_COUNT, datetime.now()))
        print(total_point / (i + 1))

    average_point = total_point / MAX_GAME_COUNT
    print(average_point)

    if average_point > 0.5:  # AlphaZeroでは0.55。マルバツだと最善同士で引き分けになるので、ちょっと下げてみました。
        update_model()


if __name__ == '__main__':
    main()
