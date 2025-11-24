import os
import numpy as np
import pandas as pd
from hmmlearn import hmm

MODEL_DIR = os.path.join('..', 'models', 'mcgill_hmm')
TEST_SONG_PATH = os.path.join('..', 'data', 'mcgill', 'metadata', 'metadata', '1069', 'bothchroma.csv')

def load_hmm_from_files(model_dir = MODEL_DIR): 
    print("Loading HMM model parameters from files...")

    pi = np.loadtxt(os.path.join(model_dir, 'mcgill_hmm_pi.txt'))
    A = np.loadtxt(os.path.join(model_dir, 'mcgill_hmm_A.txt'))
    means = np.loadtxt(os.path.join(model_dir, 'mcgill_hmm_emission_means.txt'))
    stacked_covars = np.loadtxt(os.path.join(model_dir, 'mcgill_hmm_emission_covars.txt'))

    n_states = len(pi)
    n_features = means.shape[1]
    covars = stacked_covars.reshape(n_states, n_features, n_features)

    model = hmm.GaussianHMM(
        n_components=n_states, 
        covariance_type="full", 
        init_params=""  
    )

    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars
    model.n_features = n_features
    
    print("Model successfully built.")
    return model

def get_chord_map(model_dir = MODEL_DIR): 
    idx_to_chord = {}
    with open(os.path.join(model_dir, 'mcgill_hmm_chord_map.txt'), 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            idx_to_chord[int(idx)] = label
    return idx_to_chord

if __name__ == "__main__": 
    model = load_hmm_from_files()
    chord_map = get_chord_map()

    print("Testing HMM on a sample chroma file...")
    df = pd.read_csv(TEST_SONG_PATH, header=None)
    times = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    features = df.iloc[:, 2:14].values

    mask = ~np.isnan(times)
    times = times[mask]
    features = features[mask]

    print(f"Predicting chords for {TEST_SONG_PATH}...")
    log_likelihood, state_sequence = model.decode(features, algorithm="viterbi")

    print(f"\n{'Start':<8} {'End':<8} {'Chord'}")
    print("-" * 30)

    current_state = state_sequence[0]
    start_t = times[0]
    
    for t, state in zip(times[1:], state_sequence[1:]):
        if state != current_state:
            print(f"{start_t:<8.2f} {t:<8.2f} {chord_map[current_state]}")
            current_state = state
            start_t = t
            
    print(f"{start_t:<8.2f} {times[-1]:<8.2f} {chord_map[current_state]}")