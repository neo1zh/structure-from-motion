from view import *
from match import *
from sfm import *
import numpy as np
import logging
import argparse


def run(args):

    logging.basicConfig(level=logging.INFO)
    views = create_views(args.root_dir, args.image_format)
    matches = create_matches(views)
    K = np.loadtxt(os.path.join(args.root_dir, 'images', 'K.txt'))
    print(args.is_epipolar)
    sfm = SFM(views, matches, K, args.is_epipolar, args.use_BA)
    sfm.reconstruct()


def set_args(parser):

    parser.add_argument('--root_dir', action='store', type=str, dest='root_dir',
                        help='root directory containing the images/ folder')
    parser.add_argument('--feat_type', action='store', type=str, dest='feat_type', default='sift',
                        help='type of features to be extracted [sift | surf | orb]')
    parser.add_argument('--image_format', action='store', type=str, dest='image_format', default='jpg',
                        help='extension of the images in the images/ folder')
    parser.add_argument('--is_epipolar', action='store', type=bool, dest='is_epipolar', default=False,
                        help='use epipolar to construct.')
    parser.add_argument('--use_BA', action='store', type=bool, dest='use_BA', default=False,
                        help='use epipolar to construct.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    run(args)
