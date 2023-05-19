import librosa
import os
import pickle
from tqdm import tqdm
import json
import numpy as np
import torch

from cuhkszsvc.utils.io import has_existed
from cuhkszsvc.configs.config_parse import get_wav_file_path, get_wav_path


def get_bin_index(f0, m, M, n_bins):
    """
    Args:
        f0: tensor whose shpae is (N, frame_len)
    Returns:
        index: tensor whose shape is same to f0
    """
    # Set normal index in [1, n_bins - 1]
    width = (M + 1e-7 - m) / (n_bins - 1)
    index = (f0 - m) // width + 1
    # Set unvoiced frames as 0
    index[torch.where(f0 == 0)] = 0
    # Therefore, the vocabulary is [0, n_bins- 1], whose size is n_bins
    return torch.as_tensor(index, dtype=torch.long, device=f0.device)


def get_f0_features_using_pyin(
    wave_file,
    fs,
    win_length,
    hop_length,
    f0_min,
    f0_max,
):
    y, sr = librosa.load(wave_file, sr=fs)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=f0_min, fmax=f0_max, sr=sr, win_length=win_length, hop_length=hop_length
    )
    # Set nan to 0
    f0[voiced_flag == False] = 0
    return f0


def get_log_f0(f0):
    f0[np.where(f0 == 0)] = 1
    log_f0 = np.log(f0)
    return log_f0


def extract_f0_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    f0_extractor,
    acoustic_features_name,
    fs,
    win_length,
    hop_length,
    f0_min,
    f0_max,
    using_log_f0,
):
    f0_min = librosa.note_to_hz(f0_min)
    f0_max = librosa.note_to_hz(f0_max)

    wave_dir = get_wav_path(dataset_path, dataset)
    output_dir = os.path.join(output_path, dataset, "f0/{}/{}".format(fs, f0_extractor))
    os.makedirs(output_dir, exist_ok=True)

    types = ["test"] if dataset == "m4singer" else ["train", "test"]

    for dataset_type in types:
        print("-" * 20)
        print(
            "Extracting F0 features (Using {}) for {}, {}".format(
                f0_extractor, dataset, dataset_type
            )
        )

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        if has_existed(output_file):
            continue

        # Dataset
        dataset_file = os.path.join(
            output_path, dataset, "{}.json".format(dataset_type)
        )
        if not os.path.exists(dataset_file):
            print("ERROR: File {} has not existed.".format(dataset_file))
            exit()

        with open(dataset_file, "r") as f:
            datasets = json.load(f)

        # Acoustic features
        print("Loading {} features...".format(acoustic_features_name))
        with open(
            os.path.join(
                output_path,
                dataset,
                "{}/{}/{}.pkl".format(acoustic_features_name, fs, dataset_type),
            ),
            "rb",
        ) as f:
            acoustic_features = pickle.load(f)

        # Every item is (frame_len, D)
        if acoustic_features_name == "mels":
            acoustic_features = [feat.T for feat in acoustic_features]
        print(
            "Done, #sz = {}, feats[0] is {}".format(
                len(acoustic_features), acoustic_features[0].shape
            )
        )

        f0_features = []
        for i, utt in enumerate(tqdm(datasets)):
            wave_file = get_wav_file_path(dataset, wave_dir, utt)

            # Extract
            if f0_extractor == "pyin":
                f0 = get_f0_features_using_pyin(
                    wave_file, fs, win_length, hop_length, f0_min, f0_max
                )

            if using_log_f0:
                f0 = get_log_f0(f0)

            # Check frame len with acoustic features (eg: mels)
            acoustic_feat = acoustic_features[i]
            if len(acoustic_feat) != len(f0):
                print(
                    "WARNING: [{}] {} is {}, f0 is {}\n".format(
                        wave_file, acoustic_features_name, acoustic_feat.shape, f0.shape
                    )
                )

            f0_features.append(f0)

        # Save
        with open(output_file, "wb") as f:
            pickle.dump(f0_features, f)
