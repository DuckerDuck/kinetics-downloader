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

def distribute(bins, amount):
    """Distribute amount over each bin, even if amount is not perfectly divisible 
    by the number of bins. This does not work if the remainder is larger than the 
    amount of bins"""
    bins = sorted(bins)
    minimal = int(amount / len(bins))
    missing = amount - minimal * len(bins)
    dist = {}

    for b in bins:
        dist[b] = minimal
        if missing > 0:
            dist[b] += 1
            missing -= 1

    if missing != 0:
        print(f'Warning! Could not fully distribute over bins: {missing} remaining')

    return dist


def distribute_files(args, write_to_file=True):
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
    
    dists = [{} for i in range(len(fractions) + 1)]
    # Skip last fraction (this sets the top categories to zero)
    # instead I use a uniform distribution for the last split
    for i, frac in enumerate(fractions[:-1]):
        lines = []

        videos_in_top = frac * args.total_videos 
        videos_per_cat = distribute(top_categories, videos_in_top)
        dists[i] = videos_per_cat
        lines += [f'{cat.stem}: {amount}\n' for cat, amount in videos_per_cat.items()]

        videos_not_in_top = args.total_videos - videos_in_top

        non_top_categories = [cat for cat in categories if cat not in top_categories]
        videos_per_cat = distribute(non_top_categories, videos_not_in_top)
        dists[i] = videos_per_cat
        lines += [f'{cat.stem}: {amount}\n' for cat, amount in videos_per_cat.items()]
 
        if write_to_file:
            with open(f'video_per_cat_split_{i}', 'w') as f:
                f.writelines(lines)

    # Add uniform split at the end
    videos_in_cat = int(args.total_videos / len(categories))
    lines = []
    for cat in categories:
        dists[-1][cat.stem] = videos_in_cat
        lines.append(f'{cat.stem}: {videos_in_cat}\n')

    if write_to_file:
        with open(f'video_per_cat_split_{i + 1}', 'w') as f:
            f.writelines(lines)

    return dists


def increasing_distribution(args, write_to_file=True):
    """Create distributions with increasing amount of videos,
    split uniformly over categories"""

    video_amounts = np.linspace(0, 1, args.splits) * args.total_videos

    categories, _ = stats(args, plot=False)
    dists = []
    for amount in video_amounts:
        dist = distribute(categories, int(amount))
        video_count = sum([amount for amount in dist.values()])
        print(video_count, amount)
        dists.append(dist)

    if write_to_file:
        write_distribution(dists, 'increasing_split_')

    return dists

def write_distribution(dists, prefix='video_per_cat_split_'):
    for i, dist in enumerate(dists):
        lines = [f'{cat.stem}: {amount}\n' for cat, amount in dist.items()]
        with open(prefix + str(i), 'w') as f:
            f.writelines(lines)


def read_distribution_files(i, args):
    videos_per_cat = {}
    with open(f'video_per_cat_split_{i}', 'r') as f:
        for line in f.readlines():
            split_line = line.strip().split(':')
            videos_per_cat[split_line[0]] = int(split_line[1])
    return videos_per_cat


def plot_distributions_files(args):
    f = plt.figure(figsize=(12, 8))
    has_sort = None
    sort = None

    axes = f.subplots(args.splits, 1, sharex=True)
    if args.splits == 1:
        axes = [axes]

    for s in range(args.splits):
        videos_per_cat = read_distribution_files(s, args)
        x = np.arange(len(videos_per_cat.keys()))
        values = np.array(list(videos_per_cat.values()))

        if not has_sort:
            sort = np.argsort(values)

        axes[s].set_title(f'split {s}')
        axes[s].bar(x, values[sort], width=1.0)
        axes[s].set_ylim([0, 100])
        # plt.setp(axes[s].get_xticklabels(), ha="right", rotation=90)
    plt.tight_layout()
    f.savefig('all_distributions.png')


def download_quota(args):
    """ If we want to distribute out dataset in multiple ways, we at least need to download
    the maximum amount of videos in each category for that category. This function calculates these maxes.
    """
    if args.from_file:
        dists = dists_from_file(args)
    else:
        dists = distribute_files(args, write_to_file=False)

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
        dists = distribute_files(args, write_to_file = False)

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
    parser.add_argument('--top_categories', type=int, default=50, help='Number of categories most videos should belong to.')
    parser.add_argument('--total_videos', type=int, default=4000, help='Amount of total videos to use') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_split_dir', type=str, default='split_dataset', help='Directory to store all dataset variants (where all symlinks will be created)')
    parser.add_argument('--from_file', type=bool, default=True, help='Read distribution from files')
    args = parser.parse_args()

    if args.method == 'stats':
        stats(args)
    elif args.method == 'distribute':
        distribute_files(args)
    elif args.method == 'download_quota':
        download_quota(args)
    elif args.method == 'symlink':
        create_linked_dataset(args)
    elif args.method == 'plot':
        plot_distributions_files(args)
    elif args.method == 'increase':
        increasing_distribution(args)
    else:
        print('Unknown method argument')
