## Team 4 code
    
> Note using morlet for `wavelet_energy` takes 1 min 9s and using dwt takes 0.2 for each individual
> The preprocessed files and split files are stored in split_fif folder these can be used in future without preprocessing again to load this file use `raw = mne.io.read_raw_fif(path)` then the features can be extracted as usual.