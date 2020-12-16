from pathlib import Path
import numpy as np
import argparse

def stats(args):
    output = Path(args.output_dir)
    
    if not output.is_dir():
        print(output, 'not found')
        return

    categories = list(output.iterdir())
    
    total_videos = []
    for category in categories:
        videos = list(category.iterdir())

        print(f'{category.stem}: {len(videos)}')

        total_videos.append(len(videos))

    print() 
    print(f'Number of categories: {len(categories)}')
    print(f'Total amount of videos: {np.sum(total_videos)}')
    print(f'Average amount per category: {np.mean(total_videos)}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kinetics dataset utility functions.')
    parser.add_argument('method', type=str, default='stats', help='What utility method to run: [stats]')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to kinetics download output folder')
    args = parser.parse_args()

    if args.method == 'stats':
        stats(args)
