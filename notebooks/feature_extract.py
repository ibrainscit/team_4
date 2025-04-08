
import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import skew, kurtosis
import os
import bisect

# Load the participant metadata
participants_df = pd.read_csv("TDBRAIN_participants_V2_cleaned.tsv", sep="\t")
bin_edges = [0, 6, 13, 18, 26, 41, 61]      # left edges
labels =    [0, 6, 13, 18, 26, 41, 61]


def get_gender(subject_id):
    """
    Get gender from the participants TSV file.
    Returns: 1 (Male), 0 (Female), -1 (Unknown)
    """
    row = participants_df.loc[participants_df["participants_ID"] == subject_id]
    if not row.empty:
        return int(row["gender"].values[0])  # Convert to int (0 or 1)
    return -1  # Default if not found
def get_age(subject_id):
    """
    Get age from the participants TSV file.
    Returns: Age in years
    """
    age = 0
    row = participants_df.loc[participants_df["participants_ID"] == subject_id]
    if not row.empty:
        age = int(row["age"].values[0])  # Convert to int
    idx = bisect.bisect_right(bin_edges, age) - 1
    return labels[idx]



def extract_features(vhdr_path, condition):
    """
    Extract features from preprocessed EEG data and save to CSV
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed EEG data
    condition : str
        Condition of the EEG recording (EO or EC)
    save_path : str, optional
        Path to save the extracted features CSV file
        
    Returns:
    --------
    features_df : pd.DataFrame
        DataFrame containing extracted features
    """
    # Get data and sampling frequency
    # raw = preprocess_eeg(vhdr_path)
    raw = mne.io.read_raw_fif(vhdr_path, preload=True)
    # raw.pick_types(eeg=True)  
    other_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc']
    raw.pick('eeg')  # Pick only EEG channels
    subject_id = vhdr_path.split('/')[-1].split('_')[0]
    # raw.drop_channels('Mass')  # Drop 'Mass' channel if it exists
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    # print(f"Channel names: {ch_names}")
    # Initialize feature dictionary
    features = {}
    
    features['gender'] = get_gender(subject_id)
    features['age'] = get_age(subject_id)
    # features['age']=
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    # Calculate features for each channel
    for i, ch in enumerate(ch_names):
        # Get channel data
        ch_data = data[i]
        prefix = f"{condition}_{ch.lower()}"
        # print(prefix)
        # Time domain features
        features[f'{prefix}_mean'] = np.mean(ch_data)
        features[f'{prefix}_std'] = np.std(ch_data)
        features[f'{prefix}_skew'] = skew(ch_data)
        features[f'{prefix}_kurtosis'] = kurtosis(ch_data)
        
        # Frequency domain features
        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=int(sfreq*2))
        features[f'{prefix}_psd_mean'] = np.mean(psd)
        # Band-specific FFT features
        fft_vals = np.abs(np.fft.fft(ch_data))
        fft_freqs = np.fft.fftfreq(len(ch_data), d=1/sfreq)
        
        for band, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(fft_freqs >= fmin, fft_freqs <= fmax)
            band_fft_vals = fft_vals[idx_band]
            features[f'{prefix}_{band}_fft_avg_power'] = np.mean(band_fft_vals)
        
        # Band-specific Morlet Wavelet Transform (MWT)
        valid_freqs = [f for f in freqs if f > 0]  # Ensure only positive frequencies
        
        if valid_freqs:
            mwt = mne.time_frequency.tfr_array_morlet(
                data[np.newaxis, [i], :], sfreq, freqs=valid_freqs, n_cycles=2, output='power'
            ).squeeze()
            
            for j, f in enumerate(valid_freqs):
                for band, (fmin, fmax) in bands.items():
                    if fmin <= f <= fmax:
                        features[f'{prefix}_{band}_mwt_avg_power'] = np.mean(mwt[j])
    
    # Convert to DataFrame and save as CSV
    features_df = pd.DataFrame([features])
    
    return features_df



def process_and_combine(eo_file_path, ec_file_path, output_file):
    all_features = []
    eo=False
    ec=False
    # Process EO file
    try:       
        features_eo = extract_features(eo_file_path,"EO")
        all_features.append(features_eo)
        eo=True
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    features_ec = extract_features(ec_file_path,"EC")
    all_features.append(features_ec)
    ec=True
    
    # Combine EO and EC features
    if eo and ec:
        combined_features = pd.concat(all_features,axis=1)
    # print("*****************************",combined_features.shape,"***********************************")
    # out_path = (out_dir,output_file)
    # Save combined features to a single CSV file
        combined_features.to_csv(output_file,index=False)
        print(f"Features successfully saved to {output_file}")
    # return combined_features


import os

def process_folder(source_folder, destination_folder):
    """
    Processes EO and EC files for all subjects and sessions, saving the features to CSV files.

    Args:
        source_folder (str): Path to the root folder containing subject EEG files.
        destination_folder (str): Path to the folder where CSV files will be saved.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    eo_path, ec_path = None, None
    ec_file = None  # Initialize ec_file to avoid the UnboundLocalError
    ec = False
    # Iterate through files in the source folder
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)

        if file.endswith("EC_eeg_1_eeg.fif"):
            ec_path = file_path
            ec_file = file  # Store EC file name for output filename generation
            output_filename = ec_file.replace("EC_eeg_1_eeg.fif", "eeg_combined_1.csv")
            ec=True
        if file.endswith("EC_eeg_2_eeg.fif"):
            ec_path = file_path
            ec_file = file
            ec=True
            output_filename = ec_file.replace("EC_eeg_2_eeg.fif", "eeg_combined_2.csv")
        if file.endswith("EO_eeg_1_eeg.fif") and not(ec):
            eo_path = file_path
            eo_file = file
            output_filename = ec_file.replace("EO_eeg_1_eeg.fif", "eeg_combined_1.csv")
        if file.endswith("EO_eeg_2_eeg.fif") and not(ec):
            eo_path = file_path
            eo_file = file
            output_filename = ec_file.replace("EO_eeg_2_eeg.fif", "eeg_combined_2.csv")
        # Process only when both EO and EC files are found
        if eo_path and ec_path and ec_file and ec_file:
            output_filepath = os.path.join(destination_folder, output_filename)

            print(f"Processing: \n  EO: {eo_path} \n  EC: {ec_path} \n  Output: {output_filepath}")
            process_and_combine(eo_path, ec_path, output_filepath)

            # Reset paths after processing
            eo_path, ec_path, ec_file = None, None, None



# Example usage
process_folder("/mnt/data/saikrishna/Team_4/split_fif_new/mdd","../preprocessed_data_new/mdd")
print("After")
process_folder("/mnt/data/saikrishna/Team_4/split_fif_new/healthy","../preprocessed_data_new/healthy")