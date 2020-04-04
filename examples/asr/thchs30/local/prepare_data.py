from absl import logging
import os
import sys
import codecs
from athena import get_wave_file_length
import pandas
import tensorflow as tf
import fnmatch

SUBSETS = ["train", "dev", "test"]


def convert_audio_and_split_transcript(dataset_dir, subset, out_csv_file):
    """Convert tar.gz to WAV and split the transcript.
    Args:
        dataset_dir  : the directory which holds the input dataset.
        subset       : the name of the specified dataset. e.g. dev.
        out_csv_file : the resulting output csv file.
    """
    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}".format(subset))
    src_dir = os.path.join(dataset_dir, subset)
    tar_dir = os.path.join(dataset_dir, 'data')

    if not gfile.Exists(src_dir):
        raise ValueError(src_dir, "directory is not exists.")

    files = []
    filenames = os.listdir(src_dir)
    for trans_name in fnmatch.filter(filenames, "*.wav.trn"):
        wav_name = trans_name.split('.')[0] + '.wav'
        trans_file = os.path.join(tar_dir, trans_name)
        wav_file = os.path.join(tar_dir, wav_name)
        wav_file_size = get_wave_file_length(wav_file)
        with codecs.open(trans_file, "r", "utf-8") as fin:
            lines = fin.readlines()  # read all lines
        trans = lines[0].strip('\n')
        items = trans.strip().split(" ")
        labels = ""
        for item in items:
            labels += item
        speaker = wav_name.split('_')[0]

        files.append((wav_file, wav_file_size, labels, speaker))

    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "transcript", "speakers".
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))


def processor(dircetory, subset, force_process):
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in thchs")
    if force_process:
        logging.info("force process is set to be true")
    subset_csv = os.path.join(output_dir, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the AISHELL subset {} in {}".format(subset, dataset_dir))
    convert_audio_and_split_transcript(dataset_dir, subset, subset_csv)
    logging.info("Finished processing AISHELL subset {}".format(subset))
    return subset_csv


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} dataset_dir output_dir\n'
              '    dataset_dir : directory contains thchs dataset\n'
              '    output_dir  : Athena working directory'.format(sys.argv[0]))
        exit(1)
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    for SUBSET in SUBSETS:
        processor(DATASET_DIR, SUBSET, True, OUTPUT_DIR)
