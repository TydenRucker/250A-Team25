import os
from tqdm import tqdm
import pandas as pd

ROOT_DIR = os.path.join("..", "data", "mcgill")
ANNOT_DIR = os.path.join(ROOT_DIR, "annotations", "annotations")
META_DIR = os.path.join(ROOT_DIR, "metadata", "metadata")

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

    chords = {x: i for i, x in enumerate(sorted(chordSet))}

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


if __name__ == "__main__":
    load_data()
