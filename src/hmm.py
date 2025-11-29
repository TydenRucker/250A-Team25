import os, glob
import numpy as np
import pandas as pd
from hmmlearn import hmm
from tqdm import tqdm

MODEL_DIR = os.path.join('..', 'models', 'mcgill_hmm_80_train')
ROOT_DIR = os.path.join('..', 'data', 'mcgill')
ANNOT_DIR = os.path.join(ROOT_DIR, 'annotations', 'annotations')
META_DIR = os.path.join(ROOT_DIR, 'metadata', 'metadata')
TEST_IDS_PATH = os.path.join(MODEL_DIR, 'test_ids.txt')

ENHARMONIC_MAP = { 
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 
    'Bb': 'A#', 'Cb': 'B', 'Fb': 'E', 
}

def load_hmm_from_files(model_dir): 
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
    
    print("Model built successfully.")
    return model

def load_chord_map(model_dir):
    chord_map = {} 
    label_map = {} 
    
    map_path = os.path.join(model_dir, 'mcgill_hmm_chord_map.txt')
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Could not find chord map at {map_path}")
        
    with open(map_path, 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            chord_map[int(idx)] = label
            label_map[label] = int(idx)
    return chord_map, label_map

def parse_label(label, label_map_dict): 
    if label == 'N': return 0 
    try: 
        root, quality = label.split(':') 
    except ValueError: return 0 

    root = ENHARMONIC_MAP.get(root, root) 
    
    if 'maj' in quality: simple_key = f"{root}:maj" 
    elif 'min' in quality: simple_key = f"{root}:min" 
    else: return 0 
    
    return label_map_dict.get(simple_key, 0)

def evaluate_test_set(model, test_ids, label_map):
    total_frames = 0
    correct_frames = 0
    songs_evaluated = 0

    print(f"\nEvaluating on {len(test_ids)} songs...")

    for song_id in tqdm(test_ids):
        song_id = song_id.strip() 
        
        lab_files = glob.glob(os.path.join(ANNOT_DIR, song_id, '*.lab'))
        chroma_path = os.path.join(META_DIR, song_id, 'bothchroma.csv')

        if not lab_files or not os.path.exists(chroma_path):
            print(f"Skipping {song_id}: Files missing.")
            continue

        try:
            chroma_df = pd.read_csv(chroma_path, header=None)
            times = pd.to_numeric(chroma_df.iloc[:, 1], errors='coerce').values
            features = chroma_df.iloc[:, 2:26].values
            
            valid_mask = ~np.isnan(times)
            times = times[valid_mask]
            features = features[valid_mask]

            lab_df = pd.read_csv(lab_files[0], sep='\s+', names=['start', 'end', 'label'])
            y_true = np.zeros(len(times), dtype=int)
            
            for _, row in lab_df.iterrows():
                mask = (times >= row['start']) & (times < row['end'])
                if np.any(mask):
                    state_idx = parse_label(row['label'], label_map)
                    y_true[mask] = state_idx

            _, y_pred = model.decode(features, algorithm="viterbi")

            min_len = min(len(y_true), len(y_pred))
            matches = np.sum(y_true[:min_len] == y_pred[:min_len])
            
            correct_frames += matches
            total_frames += min_len
            songs_evaluated += 1

        except Exception as e:
            print(f"Error evaluating {song_id}: {e}")
            continue

    if total_frames == 0:
        return 0.0
    
    return correct_frames / total_frames

if __name__ == "__main__": 
    try:
        model = load_hmm_from_files(MODEL_DIR)
        idx_to_chord, label_to_idx = load_chord_map(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    if not os.path.exists(TEST_IDS_PATH):
        print(f"Test IDs file not found at {TEST_IDS_PATH}")
        print("Please run the training script with --split < 1.0 first.")
        exit()
        
    with open(TEST_IDS_PATH, 'r') as f:
        test_ids = f.readlines()

    accuracy = evaluate_test_set(model, test_ids, label_to_idx)
    
    print("-" * 30)
    print(f"Frame-Level Accuracy: {accuracy * 100:.2f}%")
    print("-" * 30)