import librosa
import os
import pickle
from tqdm import tqdm
import json
import numpy as np
import torch
import sys

sys.path.append("../")
from config import dataset2wavpath, WORLD_SAMPLE_RATE, WORLD_FRAME_SHIFT,data_path


def get_bin_index(f0, n_bins=300, m = "C2", M = "C7"):
    """
    Args:
        f0: tensor whose shpae is (N, frame_len)
    Returns:
        index: tensor whose shape is same to f0
    """
    # Set normal index in [1, n_bins - 1]
    m = librosa.note_to_hz(m)
    M = librosa.note_to_hz(M)
    
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
    dataset,
    wavepath = dataset2wavpath,
    output_path = data_path,
    f0_extractor="pyin",
    acoustic_features_name = "MCEP",
    fs=WORLD_SAMPLE_RATE,
    win_length=int(0.025 * WORLD_SAMPLE_RATE),
    hop_length= int(WORLD_FRAME_SHIFT/1000 * WORLD_SAMPLE_RATE),
    f0_min = "C2",
    f0_max = "C7",
    using_log_f0 = True,
):
    f0_min = librosa.note_to_hz(f0_min)
    f0_max = librosa.note_to_hz(f0_max)

    wave_dir = wavepath[dataset]
    output_dir = os.path.join(output_path, dataset, "f0/{}/{}".format(fs, f0_extractor))
    os.makedirs(output_dir, exist_ok=True)

    types = ["test"] if dataset == "M4Singer" else ["train", "test"]

    for dataset_type in types:
        print("-" * 20)
        print(
            "Extracting F0 features (Using {}) for {}, {}".format(
                f0_extractor, dataset, dataset_type
            )
        )

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        # if has_existed(output_file):
        #     continue

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
                "{}/{}.pkl".format(acoustic_features_name, dataset_type),
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
            uid = utt["Uid"]
            if dataset == "M4Singer":
                uid = utt["Path"].split(".")[0]
            wave_file = os.path.join(wave_dir, "{}.wav".format(uid))

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

if __name__ == "__main__":
    extract_f0_features_of_dataset("M4Singer", f0_extractor="pyin", using_log_f0=True)
    extract_f0_features_of_dataset("Opencpop", f0_extractor="pyin", using_log_f0=True)