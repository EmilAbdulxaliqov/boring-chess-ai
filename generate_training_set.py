import os
import chess.pgn
import numpy as np

from state import State


def get_dataset(num_samples=None):
    X, y = [], []
    gn = 0
    values = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}

    for fn in os.listdir('data'):
        pgn = open(os.path.join('data', fn))
        while 1:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            if game is None:
                break
            res = game.headers['Result']
            value = values[res]
            board = game.board()

            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                ser = State(board).serialize()
                X.append(ser)
                y.append(value)
            print("Parsing game %d, got %d examples" % (gn, len(X)))
            if num_samples is not None and len(X) > num_samples:
                return X, y
            gn += 1
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    X, y = get_dataset(10000)
    os.makedirs('processed', exist_ok=True)
    np.savez('processed/dataset_10K.npz', X, y)
