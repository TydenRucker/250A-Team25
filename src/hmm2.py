import numpy as np
from hmmlearn import hmm

from read_data import load_data, preprocess

def calculate_probs(X, y, lengths):
    n_class = len(np.unique(y))

    initial_counts = np.zeros(n_class)

    trans_counts = np.zeros((n_class, n_class))

    cur = 0
    for l in lengths:
        song = y[cur:cur + l]
        cur += l

        initial_counts[song[0]] += 1
        for i in range(l - 1):
            trans_counts[song[i], song[i + 1]] += 1
    
    initials = initial_counts / np.sum(initials)
    trans = trans_counts / np.sum(trans, axis = 1, keepdims = True)

    # emissions
    n_feats = X.shape[1]
    means = np.zeros((n_class, n_feats))
    covars = np.zeros((n_class, n_feats, n_feats))

    for c in range(n_class):
        X_c = X[y == c]

        means[c, :] = np.mean(C_v, axis = 0)
        covars[c, :, :] = np.cov(X_c.T) + np.eye(n_feats)

    return initials, trans, means, covars


if __name__ == "__main__":
    print("loading data")
    chord_df, chroma_df = load_data()

    print("processing data for nb")
    X, y, lengths = preprocess(chord_df, chroma_df)

    X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(X, y, lengths, test_size = 0.2)

    print("training ghmm")
    initials, trans, means, covars = calculate_probs(X_train, y_train, lengths_train)

    model = hmm.GaussianHMM(n_components = 26, convariance_type = "full")

    model.init_params = ""
    model.startprob_ = initials
    model.trans_mat_ = trans
    model.means_ = means
    model.covars_ = covars

    print("predicting")
    # viterbi

    print("accuracy:", accuracy_score(y_test, y_pred))
