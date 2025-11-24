# 250A-Team25

Uses the [McGill-Billboard Songs and Chord Annotations](https://www.kaggle.com/datasets/jacobvs/mcgill-billboard/data) dataset. 

`src/process_mcgill_data.py` expects the dataset to be structured in the form
```
.
├── data
│   └── mcgill
│       ├── annotations
│       │   └── annotations
│       │       └── song_directories...
│       └── metadata
│           └── metadata
│               └── song_directories...
```
and processes the dataset to produce the model params and save them as text files inside `models/mcgill_hmm/`. 

`src/hmm.py` loads the params from these text files (so if the param files exist, there is no need to run `src/process_mcgill_data.py`) and initialises a Guassian HMM model with those params. It then runs Viterbi for a test song and prints the optimal sequence of chords for that song. 