import logging
import os
import re
from argparse import ArgumentParser
from pathlib import Path

from demo.demo import get_output
from tqdm import tqdm

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    parser = ArgumentParser(prog='test a bunch of clips')
    parser.add_argument(
        'config', metavar='config', type=str, help='model config')
    parser.add_argument(
        'checkpoint', metavar='checkpoint', type=str, help='model checkpoint')
    parser.add_argument('src_dir', type=str, help='path to the cips folder')
    parser.add_argument(
        'labels', metavar='labels', type=str, help='path to the labels')
    parser.add_argument(
        'out_dir', metavar='out', type=str, help='output dir for results')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def write_video(video,
                out_dir,
                label,
                fps=30,
                font_size=40,
                font_color='white',
                target_resolution=(None, None),
                resize_algorithm='bicubic',
                use_frames=False):
    get_output(
        video,
        out_dir,
        label,
        fps=fps,
        font_size=font_size,
        font_color=font_color,
        target_resolution=target_resolution,
        resize_algorithm=resize_algorithm,
        use_frames=use_frames)


def main():
    args = parse_args()
    assert os.path.isdir(args.src_dir), 'clips\' dir does not exist'
    if not os.path.exists(args.out_dir):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_dir, 'test-clips-logs.txt'),
        level=logging.INFO)
    model = init_recognizer(args.config, args.checkpoint, args.device)
    clips = os.listdir(args.src_dir)

    for clip in tqdm(clips, total=len(clips)):
        path_clip = os.path.join(args.src_dir, clip)
        if not os.path.isfile(path_clip):
            continue

        results = inference_recognizer(model, path_clip, args.labels)
        top_label = re.sub(r'[0-9]+', '', results[0][0])
        top_label_score = round(float(results[0][1]), 2)
        logging.info('Inspecting clip: {} '.format(path_clip))
        logging.info('Top 3 labels predicted by the model are ===> \
            (1) {}:{}, (2) {}:{}, (3) {}:{}'.format(
            top_label, top_label_score, re.sub(r'[0-9]+', '', results[1][0]),
            round(float(results[1][1]), 2), re.sub(r'[0-9]+', '',
                                                   results[2][0]),
            round(float(results[2][1]), 2)))

        write_video(path_clip, os.path.join(args.out_dir, clip),
                    top_label + str(top_label_score))


if __name__ == '__main__':
    main()
