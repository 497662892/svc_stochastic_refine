from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
import os
from pathlib import Path
import pickle

from functools import reduce

import argparse


def main(parser):
    """
    Calculates the MCD for all files in the original directory and saves the results to a pkl file.
    """
    
    original_dir = parser.parse_args().original_dir
    inferred_dir = parser.parse_args().inferred_dir
    
    results = {}
    
    # iterate over all files in the original directory
    for file in os.listdir(original_dir):
        if file.endswith('.wav'):
            original_file = Path(os.path.join(original_dir, file))
            # get the corresponding file in the inferred directory
            inferred_file = Path(os.path.join(inferred_dir, file))
            # calculate the MCD
            mcd, penalty, final_frame_number = get_metrics_wavs(original_file, inferred_file)
            # add the result to the dictionary
            results[file] = [mcd, penalty, final_frame_number]
    
    average_mcd = reduce(lambda x, y: x + y, [results[key][0] for key in results]) / len(results)
    print('Average MCD: {}'.format(average_mcd))
    
    # save results to inferred_dir as pkl
    with open(os.path.join(inferred_dir, '{}_results.pkl'.format(parser.parse_args().model)), 'wb') as f:
        pickle.dump(results, f)
    
if __name__ == '__main__':
    # add arg original_dir and inferred_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='v1')
    parser.add_argument('--original_dir', type=str, default='original')
    parser.add_argument('--inferred_dir', type=str, default='inferred')

    
    main(parser)