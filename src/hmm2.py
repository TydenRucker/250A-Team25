import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

from read_data import load_data, preprocess

def calculate_probs(X, y, lengths):
    n_class = 26

    initial_counts = np.zeros(n_class)
    trans_counts = np.zeros((n_class, n_class))

    cur = 0
    for l in lengths:
        song = y[cur:cur + l]
        cur += l

        initial_counts[song[0]] += 1
        for i in range(l - 1):
            trans_counts[song[i], song[i + 1]] += 1
    
    initials = initial_counts / np.sum(initial_counts)
    trans = trans_counts / np.sum(trans_counts, axis = 1, keepdims = True)

    # emissions
    n_feats = X.shape[1]
    means = np.zeros((n_class, n_feats))
    covars = np.zeros((n_class, n_feats, n_feats))

    for c in range(n_class):
        X_c = X[y == c]

        means[c, :] = np.mean(X_c, axis = 0)
        covars[c, :, :] = np.cov(X_c.T) + np.eye(n_feats)

    return initials, trans, means, covars

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
    initials, trans, means, covars = calculate_probs(X_train, y_train, lengths_train)

    model = hmm.GaussianHMM(n_components = 26, covariance_type = "full")

    model.startprob_ = initials
    model.transmat_ = trans
    model.means_ = means
    model.covars_ = covars

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

        correct += np.sum(pred == song_y)
        total += l
        
    print("accuracy:", correct / total)
