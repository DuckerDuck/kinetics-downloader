import pandas as pd
import argparse
import os
import shutil
import subprocess
from pandas.core.frame import DataFrame
import numpy as np
import youtube_dl
from youtube_dl.utils import YoutubeDLError
from joblib import delayed
from joblib import Parallel
import logging

logging.basicConfig(filename='failed.log', format='%(message)s')
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['label', 'youtube_id', 'time_start', 'time_end', 'split', 'is_cc']
TRIM_FORMAT = '%06d'
URL_BASE = 'https://www.youtube.com/watch?v='

VIDEO_EXTENSION = '.mp4'
VIDEO_FORMAT = 'mp4'
TOTAL_VIDEOS = 0


def create_file_structure(path, folders_names):
    """
    Creates folders in specified path.
    :return: dict
        Mapping from label to absolute path folder, with videos of this label
    """
    mapping = {}
    if not os.path.exists(path):
        os.mkdir(path)
    for name in folders_names:
        dir_ = os.path.join(path, name)
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        mapping[name] = dir_
    return mapping


def download_clip(row, label_to_dir, trim, trimmed_label_to_dir, count):
    """
    Download clip from youtube.
    row: dict-like objects with keys: ['label', 'youtube_id', 'time_start', 'time_end']
    'time_start' and 'time_end' matter if trim is True
    trim: bool, trim video to action ot not
    """

    label = row['label']
    filename = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']

    # if trim, save full video to tmp folder
    output_path = label_to_dir['tmp'] if trim else label_to_dir[label]

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][filesize <? 50M]',
    }
    
    # Don't download if the video has already been trimmed
    has_trim = False
    if trim:
        start = str(time_start)
        end = str(time_end - time_start)
        output_filename = os.path.join(trimmed_label_to_dir[label],
                                       filename + '_{}_{}'.format(start, end) + VIDEO_EXTENSION)

        has_trim = os.path.exists(output_filename)

    # Don't download if already exists
    if not os.path.exists(os.path.join(output_path, filename + VIDEO_EXTENSION)) and not has_trim:
        print('Start downloading: ', filename)  
        ydl_opts['outtmpl'] = os.path.join(output_path, '%(id)s.%(ext)s')
        
        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([URL_BASE + filename])
        except YoutubeDLError as e:
            print('Download failed for ' + filename)
            log.warning(filename)
            return False

        print('Finish downloading: ', filename)
    else:
        print('Already downloaded: ', filename)

    if trim:
        # Take video from tmp folder and put trimmed to final destination folder
        # better write full path to video


        input_filename = os.path.join(output_path, filename + VIDEO_EXTENSION)

        if has_trim:
            print('Already trimmed: ', filename)
        else:
            print('Start trimming: ', filename)
            # Construct command to trim the videos (ffmpeg required).
            command = 'ffmpeg -i "{input_filename}" ' \
                      '-ss {time_start} ' \
                      '-t {time_end} ' \
                      '-c:v libx264 -c:a copy -threads 1 -y -nostdin ' \
                      '"{output_filename}"'.format(
                           input_filename=input_filename,
                           time_start=start,
                           time_end=end,
                           output_filename=output_filename
                       )
            try:
                subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print('Error while trimming: ', filename)
                log.warning(filename)
                return False
            print('Finish trimming: ', filename)

    print('Processed %i out of %i' % (count + 1, TOTAL_VIDEOS))


def main(input_csv, output_dir, trim, num_jobs, videos_per_cat, videos_per_cat_path, seed, trim_dir):
    global TOTAL_VIDEOS

    assert input_csv[-4:] == '.csv', 'Provided input is not a .csv file'
    links_df = pd.read_csv(input_csv)
    assert all(elem in REQUIRED_COLUMNS for elem in links_df.columns.values),\
        'Input csv doesn\'t contain required columns.'


    np.random.seed(seed)

    if videos_per_cat_path != '':
        if not os.path.exists(videos_per_cat_path):
            print(f'Could not find file: {videos_per_cat_path}')
        
        videos_per_cat = {}

        with open(videos_per_cat_path, 'r') as f:
            for line in f.readlines():
                split_line = line.strip().split(':')
                videos_per_cat[split_line[0]] = int(split_line[1])

        new_links_df = []
        unique_labels = sorted(list(set(links_df['label'])))
        for label in unique_labels:
            label_rows = links_df.loc[links_df['label'] == label]
            subset_rows = label_rows[:videos_per_cat[label]]
            # subset_rows = label_rows.sample(n=videos_per_cat[label])
            print(f'Giving label {label} {videos_per_cat[label]} videos to download')
            new_links_df.append(subset_rows)
        
        # Merge back into dataframe
        links_df = pd.concat(new_links_df)


    # Creates folders where videos will be saved later
    # Also create 'tmp' directory for temporary files
    folders_names = links_df['label'].unique().tolist() + ['tmp']
    label_to_dir = create_file_structure(path=output_dir,
                                         folders_names=folders_names)

    if trim_dir is not None:
        trimmed_label_to_dir = create_file_structure(path=trim_dir, folders_names=folders_names)
    else:
        trimmed_label_to_dir = label_to_dir

    TOTAL_VIDEOS = links_df.shape[0]
    print(f'Total video count: {TOTAL_VIDEOS}')
    # Download files by links from dataframe
    Parallel(n_jobs=num_jobs)(delayed(download_clip)(
            row, label_to_dir, trim, trimmed_label_to_dir, count) for count, row in links_df.iterrows())

    # Clean tmp directory
    # shutil.rmtree(label_to_dir['tmp'])


if __name__ == '__main__':
    description = 'Script for downloading and trimming videos from Kinetics dataset.' \
                  'Supports Kinetics-400 as well as Kinetics-600.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('Path to csv file, containing links to youtube videos.\n'
                         'Should contain following columns:\n'
                         'label, youtube_id, time_start, time_end, split, is_cc'))    

    p.add_argument('--trim_dir', type=str,
                   help='Output directory where trimmed videos will be saved.\n'
                        'It will be created if doesn\'t exist')
 
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.\n'
                        'It will be created if doesn\'t exist')
    p.add_argument('--trim', action='store_true', dest='trim', default=False,
                   help='If specified, trims downloaded video, using values, provided in input_csv.\n'
                        'Requires "ffmpeg" installed and added to environment PATH')
    p.add_argument('--num-jobs', type=int, default=1,
                   help='Number of parallel processes for downloading and trimming.')
    p.add_argument('--videos-per-cat', type=int, default=-1, help='Number of videos to download for each category')
    p.add_argument('--videos-per-cat-path', type=str, default='', help='File containing the amount of videos to download for each category.')
    p.add_argument('--seed', type=int, default=42, help='Random seed for selecting videos (if not downloading all)')
    main(**vars(p.parse_args()))
