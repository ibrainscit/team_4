import joblib
import numpy as np 
import mne
import numpy as np
from scipy.stats import skew, kurtosis
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
import pandas as pd
from scipy.fftpack import fft
import pywt  # For wavelet transform
import os

import mne

def preprocess_eeg_data(vhdr_file_path, l_freq=1.0, h_freq=40.0, notch_freq=50):
    """Preprocess EEG data."""
    raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
    
    # Set EOG channels
    eog_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR']
    raw.set_channel_types({ch: 'eog' for ch in eog_channels if ch in raw.ch_names})
    
    # Apply notch filter and bandpass filter
    raw.notch_filter(freqs=[notch_freq], picks='eeg')
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg')
    
    # Set EEG reference to average
    raw.set_eeg_reference('average', projection=True)
    
    # ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    raw = ica.apply(raw)
    
    # Drop specific channels after ICA
    channels_to_drop = ['Erbs', 'OrbOcc', 'Mass']
    raw.drop_channels([ch for ch in channels_to_drop if ch in raw.ch_names])
    
    return raw



def slice_and_save_eeg(vhdr_file_path, slice_duration=60, output_dir="split_fif/mdd"):
    """Preprocess, slice, and save EEG data into 60-second segments."""
    
    # Extract the base name for saving
    base_name = os.path.splitext(os.path.basename(vhdr_file_path))[0]
    
    # Perform preprocessing
    raw = preprocess_eeg_data(vhdr_file_path)
    
    # Calculate the number of samples per slice
    samples_per_slice = int(slice_duration * raw.info['sfreq'])
    
    # Calculate the number of slices
    num_slices = int(len(raw) / samples_per_slice)
    
    # Slice and save
    for i in range(num_slices):
        start_sample = i * samples_per_slice
        end_sample = (i + 1) * samples_per_slice
        
        # Create a new raw object for the slice
        sliced_raw = raw.copy().crop(tmin=start_sample / raw.info['sfreq'], tmax=end_sample / raw.info['sfreq'], include_tmax=False)
        
        # Construct the filename for the slice
        slice_filename = f"{base_name}_{i + 1}.fif"
        
        # Save the slice in the output directory
        slice_output_path = os.path.join(output_dir, slice_filename)
        sliced_raw.save(slice_output_path, overwrite=True)
        print(f"Slice {i + 1} saved as: {slice_output_path}")

    # Handle the remaining data (if any)
    remaining_samples = len(raw) - (num_slices * samples_per_slice)
    if remaining_samples > 0:
        start_sample = num_slices * samples_per_slice
        remaining_raw = raw.copy().crop(tmin=start_sample / raw.info['sfreq'])
        remaining_filename = f"{base_name}_2.fif"
        # Save the remaining data in the output directory
        remaining_output_path = os.path.join(output_dir, remaining_filename)
        remaining_raw.save(remaining_output_path, overwrite=True)
        print(f"Remaining data saved as: {remaining_output_path}")

import os

def process_folder(source_folder, destination_folder,function_name):
    """
    Processes EO files for all subjects and sessions, saving the features to CSV files.

    Args:
        source_folder (str): Path to the root folder containing subject EEG files.
        destination_folder (str): Path to the folder where CSV files will be saved.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for sub_id in os.listdir(source_folder):
        subject_path = os.path.join(source_folder, sub_id)
        if not os.path.isdir(subject_path):
            continue
        
        for ses_id in os.listdir(subject_path):
            session_path = os.path.join(subject_path, ses_id, "eeg")
            if not os.path.isdir(session_path):
                continue
            
            for file in os.listdir(session_path):
                if file.endswith("_eeg.vhdr"):
                    inp_path = os.path.join(session_path, file)
                    output_filename = f"{file.replace('.vhdr', '.fif')}"
                    # output_filename = output_filename.replace("task-restcombined", "task-rest_combined")
                    output_path = os.path.join(destination_folder, output_filename)
                    # print(output_path)
                    function_name(inp_path)


process_folder("dataset_s/mdd","split_fif/mdd",slice_and_save_eeg)

process_folder("dataset_s/healthy","split_fif/healthy",slice_and_save_eeg)
