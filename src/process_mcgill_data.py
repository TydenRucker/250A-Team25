import os, glob
from librosa import fmt
import pandas as pd
import numpy as np

SAVE_DIR = os.path.join('..', 'models', 'mcgill_hmm')
os.makedirs(SAVE_DIR, exist_ok=True)
ROOT_DIR = os.path.join('..', 'data', 'mcgill')
ANNOT_DIR = os.path.join(ROOT_DIR, 'annotations', 'annotations')
META_DIR = os.path.join(ROOT_DIR, 'metadata', 'metadata')

CHORD_MAP = {'N': 0}
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

for i, root in enumerate(ROOTS): 
    CHORD_MAP[f"{root}:maj"] = i + 1
    CHORD_MAP[f"{root}:min"] = i + 13

def parse_label(label): 
    if label == 'N': return 0

    try: 
        root, quality = label.split(':')
    except ValueError:
        return 0
    
    if 'maj' in quality: 
        simple_key = f"{root}:maj"
    elif 'min' in quality: 
        simple_key = f"{root}:min"
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

for song_id in song_ids: 
    lab_path_pattern = os.path.join(ANNOT_DIR, song_id, '*.lab')
    lab_files = glob.glob(lab_path_pattern)
    chroma_path = os.path.join(META_DIR, song_id, 'bothchroma.csv')

    if not lab_files or not os.path.exists(chroma_path):
        continue 
    chroma_df = pd.read_csv(chroma_path)
    lab_df = pd.read_csv(lab_files[0], sep = '\t', names = ['start', 'end', 'label'])

    chroma_times = chroma_df.iloc[:, 0].values
    chroma_vals = chroma_df.iloc[:, 1:].values
    frame_states = np.zeros(len(chroma_df), dtype = int)

    for _, row in lab_df.iterrows():
        mask = (chroma_times >= row['start']) & (chroma_times < row['end'])
        if np.any(mask): 
            state_idx = parse_label(row['label'])
            frame_states[mask] = state_idx
            state_obs[state_idx].extend(chroma_vals[mask])

        if len(frame_states) > 0: 
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