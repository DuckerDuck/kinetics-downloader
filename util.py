from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def stats(args, plot=True):
    output = Path(args.dataset_dir)
    
    if not output.is_dir():
        print(output, 'not found')
        return None, None

    categories = list(output.iterdir())
    
    # Remove TMP folder
    index = -1
    for i, cat in enumerate(categories):
        if cat.stem == 'tmp':
            index = i

    if index > -1:
        del categories[index]

    
    total_videos = {}
    for category in categories:
        videos = list(category.iterdir())

        print(f'{category.stem}: {len(videos)}')

        total_videos[category] = len(videos)

    print() 
    print(f'Number of categories: {len(categories)}')
    video_values = list(total_videos.values())
    print(f'Total amount of videos: {np.sum(video_values)}')
    print(f'Average amount per category: {np.mean(video_values)}')
    
    if plot:
        # Plot distribution of videos per category
        plt.figure(figsize=(12, 8))
        x = np.arange(len(categories))
        plt.bar(x, sorted(total_videos.values()))
        plt.savefig('distribution.png')

    return categories, total_videos


def distribute(args, write_to_file=True):
    """Distributes args.total_videos over all categories. Will output args.splits number
       of distributions where there is a trade-off between the amount of videos in 
       args.top_categories and the rest"""
    categories, _ = stats(args, plot=False)

    if not categories:
        return

    if args.splits < 1:
        print('Split argument too low')
        return
    
    np.random.seed(args.seed)
    
    # Select some categories to contain the most datapoints
    top_categories = np.random.choice(categories, size=args.top_categories, replace=False)
    
    # The fraction of the videos allocated to the top categories
    fractions = np.linspace(1, 0, args.splits)
    
    dists = [{} for i in range(len(fractions))]
    for i, frac in enumerate(fractions):
        lines = []
        
        videos_in_top = frac * args.total_videos
        for cat in top_categories:
            videos_in_cat = videos_in_top / args.top_categories
            videos_in_cat = int(videos_in_cat)
            dists[i][cat.stem] = videos_in_cat
            lines.append(f'{cat.stem}: {videos_in_cat}\n')

        videos_not_in_top = (1 - frac) * args.total_videos
        for cat in categories:
            if cat in top_categories:
                continue

            videos_in_cat = videos_not_in_top / (len(categories) - args.top_categories)
            videos_in_cat = int(videos_in_cat)
            dists[i][cat.stem] = videos_in_cat
            lines.append(f'{cat.stem}: {videos_in_cat}\n')

        if write_to_file:
            with open(f'video_per_cat_split_{i}', 'w') as f:
                f.writelines(lines)
    return dists

def read_distribution_files(i, args):
    videos_per_cat = {}
    with open(f'video_per_cat_split_{i}', 'r') as f:
        for line in f.readlines():
            split_line = line.strip().split(':')
            videos_per_cat[split_line[0]] = int(split_line[1])
    return videos_per_cat


def plot_distributions_files(args):
    plt.figure(figsize=(12, 8))
    has_sort = None
    sort = None
    for s in range(args.splits):
        videos_per_cat = read_distribution_files(s, args)
        x = np.arange(len(videos_per_cat.keys()))
        values = np.array(list(videos_per_cat.values()))

        if not has_sort:
            sort = np.argsort(values)
        plt.bar(x, values[sort])

    plt.legend([f'split {i}' for i in range(args.splits)])
    plt.savefig('all_distributions.png')


def download_quota(args):
    """ If we want to distribute out dataset in multiple ways, we at least need to download
    the maximum amount of vidoes in each category for that category. This function calculates these maxes.
    """
    dists = distribute(args, write_to_file=False)

    max_per_cat = defaultdict(int)
    for dist in dists:
        for key, value in dist.items():
            max_per_cat[key] = max(max_per_cat[key], value)

    with open('download_per_cat.txt', 'w') as f:
        for key, value in max_per_cat.items():
            f.write(f'{key}: {value}\n')


    total = sum(max_per_cat.values())
    print(f'Total number of videos required to download: {total}')


def dists_from_file(args):
    dists = []
    for i in range(args.splits):
        videos_per_cat = read_distribution_files(i, args)
        dists.append(videos_per_cat)
    return dists


def create_linked_dataset(args):
    dataset = Path(args.dataset_dir)

    if not dataset.is_dir():
        print('Dataset directory not found')
        return
    
    if args.from_file:
        dists = dists_from_file(args)
    else:
        dists = distribute(args, write_to_file = False)

    parent_target = Path(args.dataset_split_dir)

    if parent_target.is_dir():
        print(f'Target directory {args.dataset_split_dir} already exists!')
        return

    parent_target.mkdir()

    for i, videos_per_cat in enumerate(dists):
        # Create directory for this split
        target = parent_target / Path(f'split_{i}')
        target.mkdir()

        for category, num_videos in videos_per_cat.items():
            target_cat = target / Path(category)
            target_cat.mkdir()

            # Find real videos to link to
            source = dataset / Path(category)

            videos = list(source.glob('*.mp4'))

            if len(videos) < num_videos:
                print(f'Warning! for category {category} we want to link {num_videos} videos but only {len(videos)} were found.')

            for video in videos[:num_videos]:
                target_video = target_cat / Path(video.name)
                target_video.symlink_to(video.resolve())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kinetics dataset utility functions.')
    parser.add_argument('method', type=str, default='stats', help='What utility method to run: [stats, distribute]')
    parser.add_argument('--dataset_dir', type=str, default='./output', help='Path to kinetics dataset folder')
    parser.add_argument('--splits', type=int, default=3, help='If redistributing data, the amount of splits between a uniform distribution and single category')
    parser.add_argument('--top_categories', type=int, default=40, help='Number of categories most videos should belong to.')
    parser.add_argument('--total_videos', type=int, default=4000, help='Amount of total videos to use') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_split_dir', type=str, default='split_dataset', help='Directory to store all dataset variants (where all symlinks will be created)')
    parser.add_argument('--from_file', type=bool, default=True, help='Read distribution from files')
    args = parser.parse_args()

    if args.method == 'stats':
        stats(args)
    elif args.method == 'distribute':
        distribute(args)
    elif args.method == 'download_quota':
        download_quota(args)
    elif args.method == 'symlink':
        create_linked_dataset(args)
    elif args.method == 'plot':
        plot_distributions_files(args)
    else:
        print('Unkown method argument')
