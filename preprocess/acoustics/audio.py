import numpy as np
import torch
import os
from tqdm import tqdm
import pickle
import json

import torchaudio
import torchaudio.functional

from cuhkszsvc.configs.config_parse import get_wav_path
from cuhkszsvc.utils.io import has_existed


def extract_audio_features(wave_file, fs):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    if sample_rate != fs:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=fs
        )
    # x: (seq,)
    x = torch.clamp(waveform[0], -1.0, 1.0).cpu().numpy()
    return x


def get_audio_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    dataset_type,
    fs,
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
    audio_features = []
    for utt in tqdm(datasets):
        uid = utt["Uid"]
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))

        if dataset == "m4singer":
            wave_file = os.path.join(wave_dir, utt["Path"])

        audio = extract_audio_features(wave_file, fs)

        audio_features.append(audio)

    return audio_features


def extract_audio_features_of_dataset(output_path, dataset_path, dataset, fs):
    output_dir = os.path.join(output_path, dataset, "audio/{}".format(fs))
    types = ["train", "test"]
    for dataset_type in types:
        print("-" * 20)
        print("Dataset: {}, {}".format(dataset, dataset_type))

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        if has_existed(output_file):
            continue

        # Extract audio features
        print("\nExtracting audio features...")
        audio_features = get_audio_features_of_dataset(
            output_path, dataset_path, dataset, dataset_type, fs
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(audio_features, f)
