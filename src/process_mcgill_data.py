import os, glob, random, argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Train HMM on McGill Dataset")
parser.add_argument(
    '--split', 
    type=float, 
    default=1.0, 
    help="Train set ratio (0.0 to 1.0). Default is 1.0 (use all data)."
)
args = parser.parse_args()

SAVE_DIR = os.path.join('..', 'models', f'mcgill_hmm_{int(args.split * 100)}_train')
os.makedirs(SAVE_DIR, exist_ok = True)
ROOT_DIR = os.path.join('..', 'data', 'mcgill')
ANNOT_DIR = os.path.join(ROOT_DIR, 'annotations', 'annotations')
META_DIR = os.path.join(ROOT_DIR, 'metadata', 'metadata')

ENHARMONIC_MAP = { 
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 
    'Bb': 'A#', 'Cb': 'B', 'Fb': 'E', 
}
CHORD_MAP = {'N': 0}
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

for i, root in enumerate(ROOTS): 
    CHORD_MAP[f"{root}:maj"] = i + 1
    CHORD_MAP[f"{root}:min"] = i + 13

def parse_label(label): 
    if label == 'N': return 0
    try: 
        root, quality = label.split(':')
    except ValueError: return 0

    root = ENHARMONIC_MAP.get(root, root)
    if 'maj' in quality: simple_key = f"{root}:maj"
    elif 'min' in quality: simple_key = f"{root}:min"
    else: return 0
    return CHORD_MAP.get(simple_key, 0)

n_states = 25
n_features = 12

start_counts = np.ones(n_states)
trans_counts = np.ones((n_states, n_states))

state_obs = {i: [] for i in range(n_states)}
song_ids = [
    d for d in os.listdir(ANNOT_DIR) if os.path.isdir(os.path.join(ANNOT_DIR, d))
]

random.seed(42) 
random.shuffle(song_ids)

n_total = len(song_ids)
n_train = int(n_total * args.split)

train_ids = song_ids[:n_train]
test_ids = song_ids[n_train:]

print(f"Total Songs: {n_total}")
print(f"Training on: {len(train_ids)} songs")
print(f"Testing on:  {len(test_ids)} songs")

if len(test_ids) > 0:
    test_file_path = os.path.join(SAVE_DIR, 'test_ids.txt')
    print(f"Saving test IDs to {test_file_path}...")
    with open(test_file_path, 'w') as f:
        for tid in test_ids:
            f.write(f"{tid}\n")

for song_id in tqdm(train_ids, desc="Processing songs"): 
    lab_path_pattern = os.path.join(ANNOT_DIR, song_id, '*.lab')
    lab_files = glob.glob(lab_path_pattern)
    chroma_path = os.path.join(META_DIR, song_id, 'bothchroma.csv')

    if not lab_files or not os.path.exists(chroma_path):
        continue 

    try:
        chroma_df = pd.read_csv(chroma_path, header=None)
        
        chroma_times = pd.to_numeric(chroma_df.iloc[:, 1], errors='coerce').values
        chroma_vals = chroma_df.iloc[:, 2:14].values
        
        valid_mask = ~np.isnan(chroma_times)
        chroma_times = chroma_times[valid_mask]
        chroma_vals = chroma_vals[valid_mask]
        
    except Exception as e:
        print(f"Error loading chroma for {song_id}: {e}")
        continue

    try:
        lab_df = pd.read_csv(lab_files[0], sep='\s+', names=['start', 'end', 'label'])
    except Exception as e:
        print(f"Error loading lab for {song_id}: {e}")
        continue

    frame_states = np.zeros(len(chroma_times), dtype=int)
    has_matches = False

    for _, row in lab_df.iterrows():
        mask = (chroma_times >= row['start']) & (chroma_times < row['end'])
        if np.any(mask): 
            has_matches = True
            state_idx = parse_label(row['label'])
            frame_states[mask] = state_idx
            state_obs[state_idx].extend(chroma_vals[mask])

    if has_matches: 
        start_counts[frame_states[0]] += 1
        for current, next_s in zip(frame_states[:-1], frame_states[1:]): 
            trans_counts[current, next_s] += 1

pi = start_counts / np.sum(start_counts)
row_sums = trans_counts.sum(axis = 1, keepdims = True)
A = np.divide(trans_counts, row_sums, out = np.zeros_like(trans_counts), where = row_sums != 0)

means = np.zeros((n_states, n_features))
covars = []

for i in range(n_states): 
    data = np.array(state_obs[i])

    if len(data) > 1: 
        means[i] = np.mean(data, axis = 0)
        covars.append(np.cov(data, rowvar = False) + 1e-3 * np.eye(n_features))
    else: 
        means[i] = np.zeros(n_features)
        covars.append(np.eye(n_features))

print("Saving HMM parameters...")
np.savetxt(os.path.join(SAVE_DIR, 'mcgill_hmm_pi.txt'), pi, fmt = '%.6f')
np.savetxt(os.path.join(SAVE_DIR, 'mcgill_hmm_A.txt'), A, fmt = '%.6f')
np.savetxt(os.path.join(SAVE_DIR, 'mcgill_hmm_emission_means.txt'), means, fmt = '%.6f')
stacked_covars = np.vstack(covars)
np.savetxt(os.path.join(SAVE_DIR, 'mcgill_hmm_emission_covars.txt'), stacked_covars, fmt = '%.6f')

with open(os.path.join(SAVE_DIR, 'mcgill_hmm_chord_map.txt'), 'w') as f:
    sorted_map = sorted(CHORD_MAP.items(), key=lambda x: x[1])
    for chord, idx in sorted_map:
        f.write(f"{idx}\t{chord}\n")

print("HMM parameters saved successfully.")