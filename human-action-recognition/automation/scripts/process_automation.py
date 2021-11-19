import os
import subprocess
from argparse import ArgumentParser

##
IN_FORMATS = ['mp4', 'avi', 'MTS']
PROCESSED_REC = '_har'
PROCESSED_DET = '_had'
##


def parse_args():
    parser = ArgumentParser(prog='automate human action recognition')
    parser.add_argument('src_dir', type=str, help='source directory')
    parser.add_argument('out_dir', type=str, help='output directory')
    parser.add_argument('checkpoint', type=str, help='the model')
    parser.add_argument('config', type=str, help='the model')
    parser.add_argument('annotations', type=str, help='annotations')
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'video'],
        default='json',
        help='output format json or video')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='recognition score threshold')
    parser.add_argument(
        '--stride',
        type=float,
        default=0,
        help=('the prediction stride equals to stride * sample_length '
              '(sample_length indicates the size of temporal window from '
              'which you sample frames, which equals to '
              'clip_len x frame_interval), if set as 0, the '
              'prediction stride is 1'))
    args = parser.parse_args()
    return args


def process(vid, out, args):
    script_path = 'demo/long_video_demo.py'
    subargs = [
        'python', script_path, args.config, args.checkpoint, vid,
        args.annotations, out, '--device',
        str(args.device), '--threshold',
        str(args.threshold), '--stride',
        str(args.stride)
    ]
    output = subprocess.run(subargs)
    print(output)


def main():
    args = parse_args()
    ext = '.json' if args.format == 'json' else '.mp4'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    in_files, out_files = os.listdir(args.src_dir), os.listdir(
        args.out_dir)  # TODO os.walk

    for to_process in [
            f for f in in_files
            if any((f.endswith(form) for form in IN_FORMATS))
    ]:
        out = os.path.splitext(to_process)[0] + PROCESSED_REC + ext
        if out in out_files:
            continue

        # TODO the prediction is quite slow. Apart from modifying the stride
        # one can also implement multipli gpu processing (4GPUs available)
        process(
            os.path.join(args.src_dir, to_process),
            os.path.join(args.out_dir, out), args)


if __name__ == '__main__':
    main()
