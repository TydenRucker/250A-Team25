import os
import numpy as np
from tqdm import tqdm
import pandas as pd

ROOT_DIR = os.path.join("..", "data", "mcgill")
ANNOT_DIR = os.path.join(ROOT_DIR, "annotations", "annotations")
META_DIR = os.path.join(ROOT_DIR, "metadata", "metadata")

chords = {
    'N': 0,
    'C:maj': 1, 
    'C:min': 2, 
    'C#:maj': 3, 
    'C#:min': 4, 
    'Db:maj': 3, 
    'Db:min': 4, 
    'D:maj': 5, 
    'D:min': 6, 
    'D#:maj': 7, 
    'D#:min': 8, 
    'Eb:maj': 7, 
    'Eb:min': 8, 
    'E:maj': 9, 
    'E:min': 10, 
    'Fb:maj': 9, 
    'Fb:min': 10, 
    'F:maj': 11, 
    'F:min': 12, 
    'F#:maj': 13, 
    'F#:min': 14, 
    'Gb:maj': 13, 
    'Gb:min': 14, 
    'G:maj': 15, 
    'G:min': 16, 
    'G#:maj': 17, 
    'G#:min': 18, 
    'Ab:maj': 17, 
    'Ab:min': 18, 
    'A:maj': 19, 
    'A:min': 20, 
    'A#:maj': 21, 
    'A#:min': 22,
    'Bb:maj': 21, 
    'Bb:min': 22, 
    'B:maj': 23, 
    'B:min': 24, 
    'Cb:maj': 23, 
    'Cb:min': 24, 
    'X': 25
}


def load_data():
    songs = list(next(os.walk(ANNOT_DIR))[1])
    songs.sort()
    
    # process chords
    chord_df = []
    chordSet = set()
    for song in songs:
        lab_path = os.path.join(ANNOT_DIR, song, "majmin.lab")
        df = pd.read_csv(lab_path, sep = "\s+", names = ["start", "end", "chord"])
        chord_df.append(df)
        chordSet.update(df["chord"])

    # chords = {x: i for i, x in enumerate(sorted(chordSet))}

    for df in chord_df:
        df["chord_id"] = df["chord"].apply(lambda x: chords[x])

    chord_df = pd.concat(chord_df, keys = songs)

    # process chroma vectors
    chroma_df = []
    cols = ["time"] + list(range(24))
    for song in tqdm(songs):
        chroma_path = os.path.join(META_DIR, song, "bothchroma.csv")
        df = pd.read_csv(chroma_path, header = None, usecols = range(1, 26), names = cols)
        chroma_df.append(df)

    chroma_df = pd.concat(chroma_df, keys = songs)

    return chord_df, chroma_df


def preprocess(chord_df, chroma_df):
    X = []
    y = []
    lengths = []

    songs = chroma_df.index.levels[0]
    for song in songs:
        chroma_data = chroma_df.loc[song].copy().sort_values('time')
        chord_data = chord_df.loc[song].copy().sort_values('start')

        # assigns each chroma vector a chord
        merged = pd.merge_asof(
            chroma_data,
            chord_data,
            left_on = "time",
            right_on = "start"
        )

        X.append(merged[list(range(24))])
        y.append(merged["chord_id"])
        lengths.append(len(merged))

    X = np.vstack(X)
    y = np.concatenate(y)

    max_vals = np.max(X, axis = 1, keepdims = True)
    max_vals[max_vals == 0] = 1

    X = X / max_vals

    return X, y, lengths


if __name__ == "__main__":
    load_data()
