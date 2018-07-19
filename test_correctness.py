from funcy        import *
from keras.models import load_model
from pathlib      import Path
from pv_mcts      import predict, pv_mcts_next_action_fn
from tictactoe    import State, popcount, random_next_action, monte_carlo_tree_search_next_action, nega_alpha_next_action


def main():
    def test_correctness(next_action):
        return ((next_action(State().next(0)) in (4,)) +
                (next_action(State().next(2)) in (4,)) +
                (next_action(State().next(6)) in (4,)) +
                (next_action(State().next(8)) in (4,)) +
                (next_action(State().next(4)) in (0, 2, 6, 8)) +
                (next_action(State().next(1)) in (0, 2, 4, 7)) +
                (next_action(State().next(3)) in (0, 4, 5, 6)) +
                (next_action(State().next(5)) in (2, 3, 4, 8)) +
                (next_action(State().next(7)) in (1, 4, 6, 8)) +
                (next_action(State().next(0).next(4).next(8)) in (1, 3, 5, 7)) +
                (next_action(State().next(2).next(4).next(6)) in (1, 3, 5, 7)))

    def first_player_point(ended_state):
        if ended_state.lose:
            return 1 if (popcount(ended_state.pieces) + popcount(ended_state.enemy_pieces)) % 2 == 1 else 0

        return 0.5

    def test_algorithm(next_actions):
        total_point = 0

        for _ in range(100):
            state = State()

            for next_action in cat(repeat(next_actions)):
                if state.end:
                    break;

                state = state.next(next_action(state))

            total_point += first_player_point(state)

        return total_point / 100

    for p in sorted(Path('./model').glob('*.h5')):
        pv_mcts_next_action = pv_mcts_next_action_fn(load_model(p))

        pv_mcts_correctness = test_correctness(pv_mcts_next_action)
        print('{:4.1f}/11 = {:.2f} pv_mcts'.format(pv_mcts_correctness, pv_mcts_correctness / 11))

    pv_mcts_next_action = pv_mcts_next_action_fn(load_model(last(sorted(Path('./model').glob('*.h5')))))

    nega_alpha_correctness = test_correctness(nega_alpha_next_action)
    print('{:4.1f}/11 = {:.2f} nega_alpha'.format(nega_alpha_correctness, nega_alpha_correctness / 11))

    pv_mcts_correctness = test_correctness(pv_mcts_next_action)
    print('{:4.1f}/11 = {:.2f} pv_mcts'.format(pv_mcts_correctness, pv_mcts_correctness / 11))

    monte_carlo_tree_search_correctness = sum(map(lambda _: test_correctness(monte_carlo_tree_search_next_action), range(100))) / 100
    print('{:4.1f}/11 = {:.2f} monte_carlo_tree_search'.format(monte_carlo_tree_search_correctness, monte_carlo_tree_search_correctness / 11))

    print(test_algorithm((pv_mcts_next_action, monte_carlo_tree_search_next_action)))
    print(test_algorithm((monte_carlo_tree_search_next_action, pv_mcts_next_action)))


if __name__ == '__main__':
    main()
