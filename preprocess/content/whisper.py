import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
import json

from . import whisper_extractor as whisper
from cuhkszsvc.configs.config_parse import get_wav_path, get_wav_file_path
from cuhkszsvc.utils.io import has_existed


def whisper_encoder(model, audio_paths):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, 3000), dtype=torch.float, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # (48000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # (80, 3000)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        batch_mel[i] = mel

    with torch.no_grad():
        # (batch, 1500, 1024)
        features = model.embed_audio(batch_mel)

    return features.cpu().detach().numpy()


def get_mapped_whisper_features(raw_whisper_features, mapping_features):
    whisper_features = []
    for index, mapping_feat in enumerate(tqdm(mapping_features)):
        sz = len(mapping_feat)

        # (1500, 1024)
        raw_feats = raw_whisper_features[index]

        feats = np.zeros((sz, raw_feats.shape[-1]), dtype=float)
        for i in range(sz):
            # Reason: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/model.py#L136
            feats[i] = raw_feats[int(i / 2)]
        whisper_features.append(feats)

    return whisper_features


def load_whisper_model(model_name):
    print("Loading Whisper Model...")
    model = whisper.load_model(model_name)
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")

    model = model.eval()
    return model


def extract_whisper_features_of_dataset(
    output_path,
    dataset_path,
    dataset,
    acoustic_features_name,
    acoustic_features_fs,
    model_name,
    whisper_seq_len,
    whisper_dim,
    batch_size,
    splits=None,
):
    output_dir = os.path.join(
        output_path,
        dataset,
        "whisper",
        "{}/{}".format(acoustic_features_name, acoustic_features_fs),
    )
    data_dir = os.path.join(output_path, dataset)
    wave_dir = get_wav_path(dataset_path, dataset)

    # Load model
    model = load_whisper_model(model_name)

    if not splits:
        splits = ["train", "test"] if dataset != "m4singer" else ["test"]

    for dataset_type in splits:
        print("-" * 20)
        print("Dataset: {}, {}".format(dataset, dataset_type))

        output_file = os.path.join(output_dir, "{}.pkl".format(dataset_type))
        if has_existed(output_file):
            continue

        with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
            datasets = json.load(f)

        # Extract raw features: (sz, 1500, 1024)
        print("\nExtracting raw whisper features...")
        whisper_features = np.zeros(
            (len(datasets), whisper_seq_len, whisper_dim), dtype=float
        )
        audio_paths = [get_wav_file_path(dataset, wave_dir, utt) for utt in datasets]

        start = 0
        end = 0
        while end < len(audio_paths):
            start = end
            end = start + batch_size
            print("{}/{}...".format(min(len(audio_paths), end), len(audio_paths)))

            whisper_features[start:end] = whisper_encoder(model, audio_paths[start:end])

        # Mapping to acoustic features' lengths
        print("\nTransform to mapping features...")
        mapping_dir = os.path.join(
            output_path,
            dataset,
            "{}/{}".format(acoustic_features_name, acoustic_features_fs),
        )
        with open(os.path.join(mapping_dir, "{}.pkl".format(dataset_type)), "rb") as f:
            mapping_features = pickle.load(f)

        # Mels: (n_mels, frame_len) -> (frame_len, n_mels)
        if acoustic_features_name == "mels":
            print("Transposing mel features...")
            mapping_features = [feat.T for feat in mapping_features]

        print(
            "Mapping to the acoustic features {}, #sz = {}, feats[0] is {}".format(
                acoustic_features_name, len(mapping_features), mapping_features[0].shape
            )
        )

        whisper_features = get_mapped_whisper_features(
            whisper_features, mapping_features
        )
        print(
            "Got whisper features, #sz = {}, feats[0] = {}".format(
                len(whisper_features), whisper_features[0].shape
            )
        )

        # Save
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(whisper_features, f)
