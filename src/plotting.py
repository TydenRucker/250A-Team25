import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from read_data import chords, load_data, preprocess

def plot_chord_distribution(y, chords):
    id_to_name = {}
    for k, v in chords.items():
        if v not in id_to_name:
            id_to_name[v] = k
        else:
            cur = id_to_name[v]
            id_to_name[v] = f"{cur.split(':')[0]}/{k}"

    # count chords
    unique, counts = np.unique(y, return_counts = True)
    total_samples = len(y)
    
    data = []
    for i, count in zip(unique, counts):
        label = id_to_name.get(i, "Unknown")
        
        # Determine Category for Color-coding
        if ":maj" in label:
            category = "Major"
        elif ":min" in label:
            category = "Minor"
        else:
            category = "N/X"
            
        data.append({
            "Chord ID": i,
            "Label": label,
            "Count": count,
            "Percentage": (count / total_samples) * 100,
            "Type": category
        })
        
    df_plot = pd.DataFrame(data)
    
    df_plot = df_plot.sort_values("Count", ascending = False)

    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    
    palette = {"Major": "#e74c3c", "Minor": "#3498db", "N/X": "#95a5a6"}
    
    ax = sns.barplot(
        data = df_plot,
        x = "Label",
        y = "Count",
        hue = "Type",
        palette = palette
    )

    # Aesthetics
    plt.title(f"Chord Class Distribution (Total Frames: {total_samples:,})")
    plt.xticks(rotation=90)
    plt.xlabel("Chord Label")
    plt.ylabel("Number of Frames")
    plt.legend(title="Chord Quality")

    # Add percentage text on top of bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + (height * 0.01),
                f'{height/total_samples:.1%}',
                ha = "center", fontsize = 9, rotation = 0
            )

    plt.tight_layout()
    plt.savefig("distribution.png",  bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y, y_pred, save_path = None):
    id_to_name = {}
    for k, v in chords.items():
        if v not in id_to_name:
            id_to_name[v] = k
        else:
            cur = id_to_name[v]
            id_to_name[v] = f"{cur.split(':')[0]}/{k}"
    labels = list(id_to_name.values())

    cm = confusion_matrix(y, y_pred, labels = list(id_to_name.keys()))

    cm_normalized = cm.astype('float') / np.sum(cm, axis = 1, keepdims = True)

    accuracy = accuracy_score(y, y_pred)

    plt.figure(figsize=(10, 10))
    sns.set_style("white")

    sns.heatmap(
        cm_normalized,
        cmap = "Blues",
        cbar = True,
        linewidths = .5,
        linecolor = 'black',
        xticklabels = labels,
        yticklabels = labels
    )

    plt.title(f"Chord Classification Confusion Matrix (Accuracy: {accuracy:.2%})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.tight_layout()

    if save_path:
        print(f"Saving confusion matrix plot to {save_path}...")
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    chord_df, chroma_df = load_data()

    X, y, lengths = preprocess(chord_df, chroma_df)

    plot_chord_distribution(y, chords)
