import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

from read_data import load_data, preprocess

def split_data_by_song(X, y, lengths, test_size = 0.2):
    song_indices = np.arange(len(lengths))

    train_idx, test_idx = train_test_split(song_indices, test_size=test_size)

    def reconstruct(indices, all_X, all_y, all_lengths):
        new_X = []
        new_y = []
        new_lengths = []

        prefix = np.cumsum([0] + list(all_lengths))

        for i in indices:
            start = prefix[i]
            end = prefix[i+1]

            new_X.append(all_X[start:end])
            new_y.append(all_y[start:end])
            new_lengths.append(all_lengths[i])

        return np.vstack(new_X), np.concatenate(new_y), new_lengths

    X_train, y_train, len_train = reconstruct(train_idx, X, y, lengths)
    X_test, y_test, len_test = reconstruct(test_idx, X, y, lengths)

    return X_train, X_test, y_train, y_test, len_train, len_test


if __name__ == "__main__":
    print("loading data")
    chord_df, chroma_df = load_data()

    print("processing data for ghmm")
    X, y, lengths = preprocess(chord_df, chroma_df)

    X_train, X_test, y_train, y_test, lengths_train, lengths_test = split_data_by_song(X, y, lengths, test_size = 0.2)

    print("training ghmm")

    model = hmm.GaussianHMM(
        n_components = 26, 
        covariance_type = "diag",
        n_iter = 50,
        tol = 0.01,
        verbose = True
    )

    model.fit(X_train, lengths = lengths_train)

    train_states = model.predict(X_train, lengths = lengths_train)
    hmm_to_true_map = {}

    for state_id in range(model.n_components):
        indices = np.where(train_states == state_id)[0]

        if len(indices) > 0:
            true_labels = y_train[indices]

            most_common_label = np.bincount(true_labels).argmax()
            hmm_to_true_map[state_id] = most_common_label
        else:
            hmm_to_true_map[state_id] = 0

    print("predicting")
    # viterbi
    correct = 0
    total = 0
    
    cur = 0
    for l in tqdm(lengths_test):
        song_X = X_test[cur : cur + l]
        song_y = y_test[cur : cur + l]
        cur += l

        pred = model.predict(song_X)

        pred = np.array([hmm_to_true_map[s] for s in pred])

        correct += np.sum(pred == song_y)
        total += l
        
    print("accuracy:", correct / total)
