import numpy as np
import torch
import os
from tqdm import tqdm
import pickle
import json

import torchaudio
import torchaudio.functional
import torchaudio.transforms

from cuhkszsvc.configs.config_parse import get_wav_path
from cuhkszsvc.utils.io import has_existed
from cuhkszsvc.configs.config_parse import get_wav_file_path


def normalize(mel, min_level_db):
    mel_norm = torch.clamp((mel + min_level_db) / min_level_db, 0.0, 1.0)
    return mel_norm.cpu().numpy()


def extract_mel_features(
    wave_file,
    fs,
    win_length,
    hop_length,
    n_fft,
    n_mels,
):
    # waveform:(1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    if sample_rate != fs:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=fs
        )
    # x: (seq,)
    x = torch.clamp(waveform[0], -1.0, 1.0)
    # mel: (n_mels, bins)
    with torch.no_grad():
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=fs,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            f_min=20,
            f_max=fs / 2,
            n_mels=n_mels,
            power=1.0,
            normalized=True,
        )(x)
        # amplitude_to_db
        mel = 20 * torch.log10(torch.clamp(mel, min=1e-5)) - 20

    return mel


def get_mel_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    dataset_type,
    fs,
    win_length,
    hop_length,
    n_fft,
    n_mels,
):
    data_dir = os.path.join(output_path, dataset)
    wave_dir = get_wav_path(dataset_path, dataset)

    # Dataset
    dataset_file = os.path.join(data_dir, "{}.json".format(dataset_type))
    if not os.path.exists(dataset_file):
        print("File {} has not existed.".format(dataset_file))
        return None

    with open(dataset_file, "r") as f:
        datasets = json.load(f)

    # Extract
    mel_features = []
    for utt in tqdm(datasets):
        wave_file = get_wav_file_path(dataset, wave_dir, utt)
        mel = extract_mel_features(wave_file, fs, win_length, hop_length, n_fft, n_mels)

        mel_features.append(mel)

    return mel_features


def extract_mel_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    fs,
    win_length,
    hop_length,
    n_fft,
    n_mels,
    min_level_db,
):
    output_dir = os.path.join(output_path, dataset, "mels/{}".format(fs))
    types = ["train", "test"]
    for dataset_type in types:
        print("-" * 20)
        print("Dataset: {}, {}".format(dataset, dataset_type))

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        if has_existed(output_file):
            continue

        # Extract mel features
        print("\nExtracting Mel featuers...")
        mel_features = get_mel_features_of_dataset(
            output_path,
            dataset_path,
            dataset,
            dataset_type,
            fs,
            win_length,
            hop_length,
            n_fft,
            n_mels,
        )

        # Normalize mel features
        print("\nNormalizing mel features...")
        mel_norm_features = [normalize(mel, min_level_db) for mel in mel_features]

        # Save
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(mel_norm_features, f)
