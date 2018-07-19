import numpy as np
import pickle

from funcy           import *
from keras.callbacks import LearningRateScheduler
from keras.models    import load_model, save_model
from pathlib         import Path
from pv_mcts         import to_x
from operator        import getitem


def load_data():
    def load_datum(path):
        with path.open(mode='rb') as f:
            return pickle.load(f)

    states, y_policies, y_values = zip(*map(load_datum, tuple(sorted(Path('./data').glob('*.pickle')))[-5000:]))

    return map(np.array, (tuple(map(to_x, states)), y_policies, y_values))


def main():
    xs, y_policies, y_values = load_data()

    model_path = last(sorted(Path('./model/candidate').glob('*.h5')))
    model = load_model(model_path)

    model.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer='adam')
    model.fit(xs, [y_policies, y_values], 100, 100,
              callbacks=[LearningRateScheduler(partial(getitem, tuple(take(100, concat(repeat(0.001, 50), repeat(0.0005, 25), repeat(0.00025))))))])

    save_model(model, model_path.with_name('{:04}.h5'.format(int(model_path.stem) + 1)))


if __name__ == '__main__':
    main()
