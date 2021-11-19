import logging
import os
import os.path as osp
import re
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from itertools import repeat
from math import floor
from multiprocessing import Pool, cpu_count

import numpy as np
from moviepy.editor import VideoFileClip

sys.path.append('/mmaction2/human-action-recognition/')  # noqa
import har.tools.helpers as helpers  # noqa isort:skip

# * This script processes the raw dataset to an acceptable format for mmaction2
# Moreover, the script takes care of:
#   1) Videos with wrong annotations
#   2) Videos with multiple annotations (not present in the xml files)
#
# the old library that was used to cut clips
# * from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#
# We use 15 videos out of a total of 98 picked before hand for pure testing.
# The rest 83 videos are split according to an 80/20 rule.
# We end up with:
#   - Train: 70% of the total dataset
#   - Validation: 15% (in the script here refered as Test)
#   - Test: 15% (in the script here refered as Validation)
#
# Types of files that this script processes:
#   1) videos .mp4, .MTS
#   2) xml annotations .eaf
#
# * Based on the structure required by mmaction2:
#   https://mmaction2.readthedocs.io/en/latest/tutorials/3_new_dataset.html
# The two main dataset formats are Videos & RawFrames.
# This script focuses on the former. Custom formats are also possible.
#
# Link to dataset:
#   https://dshs-koeln.sciebo.de/index.php/s/Y67O1kG1lEtXuV9?path=%2F
#
# * The input that this script takes
# The raw videos & annotations must be in the same folder,
# e.g. for "Ausstehende BAST Ratings\Deutsche BAST"
#   - remove the 'avatar' videos that do not have annotations (ending in _2)
#   - keep only the corresponding .eaf files
#   In the end you should be left with the following
#   videos = ['#000_1'  '#005_1'  '#010_1' ... '#041_1'  '#042_1'  '#043_1']
#   annotations = ['GER_000_1.eaf'  'GER_005_1.eaf'  'GER_010_1.eaf' ...
#                           'GER_041_1.eaf' 'GER_042_1.eaf'  'GER_043_1.eaf']
#
# Annotation files that had to be manually edited
#   - V_2_std.eaf
#       <ALIGNABLE_ANNOTATION ANNOTATION_ID="a16" TIME_SLOT_REF1="ts34"
#                   TIME_SLOT_REF2="ts39"> -> FIX ID, annotation_id = "a7"
#
# ! In some videos the candidate is not performing the action but the
# ! annotations are still there,
# ! e.g. BAST_Auswertungen_Clara\Standard_Japaner\V_24_std
#
# ! TODO: .MTS videos are not cropped correctly. Probably some problem when
# TODO: saving as mp4. So you see many such videos containing the next
# TODO: annotation even though the timestamps are perfectly fine. As it
# TODO: is the videos seem to cover more than 10s. Maybe use the
# TODO: script resize_video.py? maybe check the original videos with
# TODO: check_videos.py
#
# TODO: Checkout the Errors in the log file

TANZ_BASE_PATH = 'data/tanz_base'
TANZ_EVAL_PATH = 'data/tanz_evaluation'
BASE_ANN_PATH = ('human-action-recognition/har/annotations/BAST/base/'
                 'tanz_annotations.txt')
EVAL_ANN_PATH = ('human-action-recognition/har/annotations/BAST/eval/'
                 'tanz_annotations_42.txt')
# hard cut-off regardless of window or clip size
MAX_NO_CLIPS = 20
LABEL_TO_NUMBER = {}
# the pure testing set
VALIDATION_SET = [
    '#000_1.mp4',
    '#043_1.mp4',
    '2019_01_1.mp4',
    '2019_04_1.mp4',
    '2019_15_1.mp4',
    '#027_1.mp4',  # up to here, there are no evaluation ann
    # * 10 videos in total for the evaluation annotations
    '#003_1.mp4',
    '#036_1.mp4',
    '#039_1.mp4',
    '#108_1.mp4',
    '2019_33.MTS',
    'j#03_1.mp4',
    'j#04_1.mp4',
    '2019_18.MTS',
    '2019_29.MTS',
    '2019_30.MTS'
]
WRONG_ANN = {
    'j#07_1.mp4': 20,
    'j#11_1.mp4': 10,
    'j#13_1.mp4': 10,
    'j#17_1.mp4': 14,
    'j#19_1.mp4': 10,
    'j#06_1.mp4': 15,
    'j#24_1.mp4': 10,
    'j#18_1.mp4': 17,
    'j#09_1.mp4': 8,
    'j#08_1.mp4': 15,
    'j#12_1.mp4': 8,
    'j#14_1.mp4': 9,
    # Validation
    '2019_18.MTS': -18,
    '2019_29.MTS': -19,
    '2019_30.MTS': -17,
}
# {video:{the_specific_label_with_double_anntations:
#                           the_second_annotation}, ...}
# only the case with swing_upper_body & tiptoe
DOUBLE_ANN = {
    'j#03_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#06_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#07_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#10_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#12_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#14_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#15_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#16_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#17_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#19_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#21_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#22_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
    'j#23_1.mp4': {
        'swing_upper_body': 'tiptoe'
    },
}
EVALUATION_ANN = {
    # first find the specific movement
    # then check the evaluation and get the annotation
    # e.g. EVAL_ANN['WAL-floorpattern']['straight more than curved']
    #   will give as annotation "walk_more_straight"
    'WAL-floorpattern': {
        'straight': 'not_enough_samples',  # 'walk_straight',
        'straight more than curved': 'walk_straight_more',
        'curved more than straight': 'walk_curved_more',
        'curved': 'walk_curved',
        '?': 'not_found',
    },
    'RUN-floorpattern': {
        'straight': 'not_enough_samples',  # 'run_straight',
        'straight more than curved': 'run_straight_more',
        'curved more than straight': 'run_curved_more',
        'curved': 'run_curved',
        '?': 'not_found',
    },
    'JUM-emphasis': {
        'upwards': 'jump_emphasis_upward',
        'downwards': 'not_enough_samples',  # 'jump_emphasis_downwards',
        'forward': 'jump_emphasis_forward',
        'sidewards': 'not_enough_samples',  # 'jump_emphasis_sidewards',
        '?': 'not_found',
    },
    'JUM-time-in-air': {
        'short': 'jump_time_short',
        'long': 'jump_time_long',
        '?': 'not_found',
    },
    'STA-bodyinvolvement': {
        'isolated': 'stamp_body_isolated',
        'whole body': 'stamp_body_whole',
        '?': 'not_found',
    },
    'STA-strength': {
        'no strength': 'stamp_strength_none',
        'little strength': 'stamp_strength_little',
        'medium strength': 'stamp_strength_medium',
        'maximum strength': 'not_enough_samples',  # 'stamp_strength_max',
        '?': 'not_found',
    },
    'CON-kinesphere': {
        'narrow': 'contract_narrow',
        'narrow more than wide': 'contract_narrow_more',
        'wide more than narrow': 'not_enough_samples',  # 'contract_wide_more',
        'wide': 'not_enough_samples',  # 'contract_wide',
        '?': 'not_found',
    },
    'EXP-kinesphere': {
        'narrow': 'not_enough_samples',  # 'expand_narrow',
        'narrow more than wide': 'expand_narrow_more',
        'wide more than narrow': 'expand_wide_more',
        'wide': 'expand_wide',
        '?': 'not_found',
    },
    'COEX-emphasis': {
        'no emphasis': 'con_exp_no_emphasis',
        'contracting': 'con_exp_emphasis_contracting',
        'expanding': 'con_exp_emphasis_expanding',
        '?': 'not_found',
    },
    'COEX-headintegration': {
        'desintegrated': 'hard',  # 'con_exp_head_desintegrated',
        'integrated': 'hard',  # 'con_exp_head_integrated',
        '?': 'not_found',
    },
    'TOE-balance': {
        'unstable': 'tiptoe_unstable',
        'unstable more than stable':
        'not_enough_samples',  # tiptoe_unstable_mor
        'stable more than unstable': 'tiptoe_stable_more',
        'stable': 'tiptoe_stable',
        '?': 'not_found',
    },
    'SWI-flow': {
        'bound': 'swing_flow_bound',
        'bound more than free': 'swing_bound_more',
        'free more than bound': 'swing_free_more',
        'free': 'swing_free',
        '?': 'not_found',
    },
    'SWI-bodyinvolvement': {
        'isolated': 'not_enough_samples',  # 'swing_isolated',
        # 'swing_isolated_more',
        'isolated more than whole body': 'not_enough_samples',
        # 'swing_wholebody_more',
        'wholebody more than isolated': 'not_enough_samples',
        # 'swing_wholebody_more',
        'whole body more than isolated': 'not_enough_samples',
        'wholebody': 'not_enough_samples',  # 'swing_wholebody',
        '?': 'not_found',
    },
    'SWI-headintegration': {
        'desintegrated': 'hard',  # 'swing_head_desintegrated',
        'integrated': 'hard',  # 'swing_head_integrated',
        '?': 'not_found',
    },
    'SPI-flow': {
        'bound': 'not_enough_samples',  # 'spin_bound',
        'bound more than free': 'spin_bound_more',
        'free more than bound': 'spin_free_more',
        'free': 'spin_free',
        '?': 'not_found',
    },
    'SPI-continuity': {
        'one by one': 'spin_continuity_single',
        'discontinued': 'spin_continuity_discontinued',
        'continued': 'spin_continuity_continued',
        '?': 'not_found',
    },
    'SPI-orientation': {
        'fixates': 'imbalanced',  # 'spin_orientation_fixate',
        'does not fixate': 'imbalanced',  # 'spin_orientation_fixate_not',
        '?': 'not_found',
    },
    'SPI-acceleration': {
        'no acceleration': 'spin_acceleration_none',
        'acceleration': 'spin_acceleration',
        '?': 'not_found',
    },
    'FAL-flow': {
        'sits/lies down': 'fall_lie_down',
        'free falling': 'fall_free',
        'falls in steps': 'not_enough_samples',  # 'fall_steps',
        '?': 'not_found',
    },
    'FAL-endposition': {
        'sitting': 'fall_pos_sit',
        'more thansittingless thanlying':
        'not_enough_samples',  # fal_pos_lie_ha
        'lying': 'fall_pos_lie',
        '?': 'not_found',
    },
}
GERMAN_TO_ENGLISH = {
    'Gehen': 'walk',
    'Laufen': 'run',
    'Springen': 'jump',
    'Stampfen': 'stamp',
    'Stanpfen': 'stamp',
    'Zusammenziehen/Ausdehnen': 'contract_expand',
    'Zusammenziehen / Ausdehen': 'contract_expand',
    'Zusammenziehen / Ausdehnen': 'contract_expand',
    'Zusammenziehen/ Ausdehnen': 'contract_expand',
    'Zusammenziehen /Ausdehnen': 'contract_expand',
    'Ballenstand': 'tiptoe',
    'Ballenstände': 'tiptoe',
    'Hochzehenstand': 'tiptoe',
    'Schwünge': 'swing_upper_body',
    'Schwingen': 'swing_upper_body',
    'schwingen': 'swing_upper_body',
    'Drehen': 'rotate',
    'Fallen': 'fall'
}


def create_annotation_file(base_path, classes):
    with open(osp.join(base_path, 'tanz_annotations.txt'), 'w') as ann:
        for tup in zip(classes, range(len(classes))):
            cls = tup[0].replace('_', '-')
            ann.write(f'{tup[1]} {cls}\n')


def generate_structure(ann_type):
    """Generate an empty structure for the base or evaluation BAST clips based
    on the VideoDataset of mmaction2 See above link for more details."""

    classes_base = [
        label.replace('-', '_')
        for label in helpers.bast_annotations_to_list(BASE_ANN_PATH)
    ]
    classes_eval = [
        label.replace('-', '_')
        for label in helpers.bast_annotations_to_list(EVAL_ANN_PATH)
    ]

    path = TANZ_BASE_PATH if ann_type == 'base' else TANZ_EVAL_PATH
    classes = classes_base if ann_type == 'base' else classes_eval

    print(f'Structure for {ann_type} tanz dataset not found. Generating...')
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    for dIr in ['annotations', 'videos_train', 'videos_val']:
        Path(osp.join(path, dIr)).mkdir(parents=True, exist_ok=True)
        if dIr == 'annotations':
            create_annotation_file(osp.join(path, dIr), classes)
        else:
            for cl in classes:
                Path(osp.join(path, dIr, cl)).mkdir(
                    parents=True, exist_ok=True)

    open(osp.join(path, 'tanz_train_list_videos.txt'), 'w').close()
    open(osp.join(path, 'tanz_val_list_videos.txt'), 'w').close()


def reset_structure(ann_type):
    """Delete the current generated dataset."""
    path = TANZ_BASE_PATH if ann_type == 'base' else TANZ_EVAL_PATH
    if not osp.exists(path):
        return

    for split in ['videos_train', 'videos_val']:
        path_to_split = osp.join(path, split)
        if not osp.exists(path_to_split):
            return

        for label in os.listdir(path_to_split):
            label_path = osp.join(path_to_split, label)
            for clip in os.listdir(label_path):
                os.unlink(osp.join(label_path, clip))
            os.rmdir(label_path)

    open(osp.join(path, 'tanz_train_list_videos.txt'), 'w').close()
    open(osp.join(path, 'tanz_val_list_videos.txt'), 'w').close()
    open(osp.join(path, 'annotations', 'tanz_annotations.txt'), 'w').close()
    print(f'# {ann_type}-tanz dataset successful reseted')


def cleanup():
    """The clip creation library generates intermediate clips.

    This function removes those clips after the script finishes execution.
    """
    for f in os.listdir():
        if f[-4:] == '.mp4':
            os.unlink(f)


def get_log_file_name(no_vid, args):
    file_name = 'tanz_' + args.ann_type + '_' + str(no_vid) + 'vid_' + str(
        args.test_split) + 'split_' + str(args.clip_length) + 's_clips_' +\
        str(args.sliding_window) + 's_sliding_' + 'V1.log'
    i = 2
    while osp.isfile(file_name):
        file_name = file_name[:-5] + str(i) + '.log'
        i += 1
    return file_name


def get_base_class(label):
    base_labels = [
        label.split('-')[0]
        for label in helpers.bast_annotations_to_list(BASE_ANN_PATH)
    ]

    base_label = None
    for bl in base_labels:
        if bl == 'rotate':
            bl = 'spin'
        if bl in label:
            base_label = bl
            break

    if (base_label is None) | (base_label == 'contract'):
        base_label = 'contract_expand'
    elif base_label == 'spin':
        base_label = 'rotate'
    elif base_label == 'swing':
        base_label = 'swing_upper_body'
    return base_label


def parse_args():
    parser = ArgumentParser(prog='generate BAST VideoDataset. Base & eval ann')
    parser.add_argument(
        'src_dir',
        type=str,
        help='source video directory. Contains video and annotations')
    parser.add_argument(
        '--ann-type',
        default='base',
        choices=['base', 'eval'],
        help='type of annotations for which to generate the dataset')
    parser.add_argument(
        '--test-split',
        type=helpers.float_parser,
        default='0.18',
        required=False,
        help='train/validation ratio. Give for validation')
    parser.add_argument(
        '--clip-length',
        type=int,
        default=10,
        required=False,
        help='length of each clip; default is best')
    parser.add_argument(
        '--sliding-window',
        type=int,
        default=5,
        required=False,
        help='sliding window of generated clips; default is best')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=(cpu_count() - 2 or 1),
        help='number of processes used')
    parser.add_argument(
        '--test-dataset',
        action='store_true',
        help='generate test dataset with the videos specified in the '
        'VALIDATION_SET dict')
    args = parser.parse_args()
    return args


def merge_train_test(path, ann_type):
    import shutil

    # copy clips
    val_path = osp.join(path, 'videos_val')
    train_path = osp.join(path, 'videos_train')
    for cls in os.listdir(val_path):
        cls_val = osp.join(val_path, cls)
        cls_train = osp.join(train_path, cls)

        for clip in os.listdir(cls_val):
            shutil.move(osp.join(cls_val, clip), cls_train)

    # copy clips list
    with open(osp.join(path, 'tanz_val_list_videos.txt'), 'r') as file:
        val_ann = [line for line in file]
    assert len(val_ann) > 0

    with open(osp.join(path, 'tanz_train_list_videos.txt'), 'a') as file:
        for line in val_ann:
            line = line.replace('videos_val', 'videos_train')
            file.write(line)

    # restructure
    shutil.rmtree(val_path)
    shutil.rmtree(osp.join(path, 'annotations'))
    os.unlink(osp.join(path, 'tanz_val_list_videos.txt'))
    os.rename(train_path, osp.join(path, 'clips_eval'))
    os.rename(
        osp.join(path, 'tanz_train_list_videos.txt'),
        osp.join(path, f'tanz_{ann_type}_test_list_videos.txt'))

    log = f'Merged videos_val with videos_train @{path}'
    print(log)
    logging.info(log)


def gen_id(path_to_save):
    clip_ids = list(map(int, [clip[:-9] for clip in os.listdir(path_to_save)]))
    clip_id = str(max(clip_ids) + 1) if len(clip_ids) > 0 else '0'
    # concurent threads can access the same id so we add some random chars
    return clip_id + '_' + str(uuid.uuid4()).split('-')[1]


def save_annotation(clip_path, label, train_test, path):
    """Store the corresponding annotation after the clip has been created."""
    train_file = 'tanz_train_list_videos.txt'
    test_file = 'tanz_val_list_videos.txt'
    train_test = train_file if train_test == 'videos_train' else test_file
    with open(osp.join(path, train_test), 'a') as ann_file:
        ann_file.write('/'.join(clip_path.split('\\')[-2:]) + ' ' +
                       str(LABEL_TO_NUMBER[label]))
        ann_file.write('\n')


def check_faulty_annotations(video, start, finish):
    """Fix faulty annotations."""
    key = video.split('/')[-1]
    return start - WRONG_ANN.get(key, 0), finish - WRONG_ANN.get(key, 0)


def check_double_annotations(video, label):
    """ Check if a video has more than one label for one particular clip
        Returns: a list of labels
    """
    key = video.split('/')[-1]
    # no video
    if type(DOUBLE_ANN.get(key, [])) == list:
        return [label]
    # video exists but no label
    if type(DOUBLE_ANN.get(key, []).get(label, [])) == list:
        return [label]
    return [label, DOUBLE_ANN[key][label]]


def save_clips(video, label, start, finish, train_test, clip_length,
               sliding_window, ann_type):
    """Save a bunch of sub-clips from an annotatated clip based on clip_length.

    & sliding_window.

    Start & finish time have been rounded for best split.
    """

    start, finish = floor(start), floor(finish)
    start, finish = check_faulty_annotations(video, start, finish)
    path = TANZ_BASE_PATH if ann_type == 'base' else TANZ_EVAL_PATH
    if ann_type == 'base':
        paths_to_save = [
            osp.join(path, train_test, label)
            for label in check_double_annotations(video, label)
        ]
    else:
        paths_to_save = [osp.join(path, train_test, label)]

    if label[:4] == 'fall':
        # 'fall', 'fall_lie_down', 'fall_free'
        current_clip_path = osp.join(paths_to_save[0],
                                     gen_id(paths_to_save[0]) + '.mp4')
        with VideoFileClip(video) as v:
            try:
                clip = v.subclip(start, finish)
                clip.write_videofile(
                    current_clip_path, logger=None, audio_codec='aac')
            except OSError as e:
                # MoviePy error:
                # b'[mov,mp4,m4a,3gp,3g2,mj2 @ 0x68d46c0] moov atom not found
                log = (f'! Corrupted Video: {video} | Label: {label}'
                       f' | Start: {start} | Finish: {finish} | Error: {e}')
                print(log)
                logging.info(log)
                logging.exception('Corrupted Video')
                return
            except (IOError, ValueError) as e:
                # TODO: fall annotations out of bounds, wrong annotation
                log = (f'! Start & end out of bounds video: {video}'
                       f' | Label :{label} | Start: {start} | End: {finish}')
                print(log, e)
                logging.info(log)
                logging.exception('Start & End out of bounds')
                return
        save_annotation(current_clip_path, label, train_test, path)
    else:
        for path_to_save in paths_to_save:
            for i in range(MAX_NO_CLIPS):
                current_clip_path = osp.join(path_to_save,
                                             gen_id(path_to_save) + '.mp4')
                clip_s = start + sliding_window * i
                clip_f = clip_s + clip_length

                if clip_f > finish:
                    # TODO: maybe still use the clip if >6 sec left?
                    break

                with VideoFileClip(video) as v:
                    clip = v.subclip(clip_s, clip_f)
                    try:
                        clip.write_videofile(
                            current_clip_path, logger=None, audio_codec='aac')
                    except OSError as e:
                        log = (
                            f'! Corrupted Video: {video} | Label: {label}'
                            f' | Start: {start} | Finish: {finish} | Error:{e}'
                        )
                        print(log)
                        logging.info(log)
                        logging.exception('Corrupted Video')
                        continue
                save_annotation(current_clip_path, label, train_test, path)


def get_video_annotation(video, annotations):
    """Given a video, get its corresponding xml annotation file.

    This func assumes that all videos and annotations are in the same folder.
    Moreover, it is strictly based on the namings of these ann & videos.
    """
    if video.find('2019') != -1:
        # Ausstehende Bast Ratings/ Japaner 1 bis 15
        try:
            curr_ann = next(annotation for annotation in annotations
                            if video[:-4] == annotation[:-4])
        except StopIteration:
            # Testungen 16-33 (with ratings in Auswertungen Clara)
            try:
                curr_ann = next(annotation for annotation in annotations
                                if video[5:7] == annotation[0:2])
            except StopIteration:
                pass
    elif (video == 'JPN_34_1.mp4') | (video == 'JPN_35_1.mp4'):
        # Ausstehende Bast Ratings/ Japaner 16-35 are an exception
        # due to faulty namings
        if video == 'JPN_34_1.mp4':
            curr_ann = '2019_34_1.eaf'
        else:
            curr_ann = '2019_35_1.eaf'
    elif video[:2] == 'j#':
        # root Japan (with ann in BAST_Auswertungen_Clara\Standard_Japaner)
        # half of the videos here are faulty
        j_ann = [
            annotation for annotation in annotations if annotation[0:2] == 'V_'
        ]
        curr_ann = next(
            annotation for annotation in j_ann if int(video[2:4]) == int(
                re.search(r'\d+',
                          annotation[2:4]).group()))  # check if the ids match
    else:
        try:
            # Ausstehende Bast Ratings/ Deutsche Bast
            curr_ann = next(annotation for annotation in annotations
                            if video[1:6] == annotation.strip('GER_.eaf'))
        except StopIteration:
            # root German (with ann in BAST_Auswertungen_Clara\Standard_Spoho)
            curr_ann = next(annotation for annotation in annotations
                            if video[1:4] == annotation[1:4])
    return curr_ann


def split_video_to_clips_base(index, video, annotations, test_split,
                              clip_length, sliding_window, ann_type):
    """Splits a video to clips based on its annotations, the clip length and
    the sliding window.

    Only for the base annotations.
    """

    log = f'\n## Processing video #{index}: {video}...'
    logging.info(log)
    print(log)

    # build xml tree
    tree = ET.parse(annotations)
    root = tree.getroot()

    # get annotations with their corresponding timestamp markers
    # [{'Gehen': 'ts1,ts3'}, ..., {'Fallen': 'ts53,ts56'}]
    splits = []
    base_ann_ids = ['a' + str(i) for i in range(0, 10)]
    for base_annotation in root.iter('ALIGNABLE_ANNOTATION'):
        if base_annotation.attrib['ANNOTATION_ID'] not in base_ann_ids:
            continue
        time_ref1 = base_annotation.attrib['TIME_SLOT_REF1']
        time_ref2 = base_annotation.attrib['TIME_SLOT_REF2']
        annotation = base_annotation[0].text.strip()
        splits.append({annotation: time_ref1 + ',' + time_ref2})

    # get all timestamps
    # [{'ts1': '3320'}, ..., {'ts58': '235798'}]
    time_stamps = []
    for time_slot in root.iter('TIME_SLOT'):
        time_stamps.append(
            {time_slot.attrib['TIME_SLOT_ID']: time_slot.attrib['TIME_VALUE']})

    # make sure to get different results in the different processes
    np.random.seed()
    train_test = 'videos_train' if np.random.choice(
        [0, 1], p=[1 - test_split, test_split]) == 0 else 'videos_val'
    log = f'### Saving clips as {train_test[7:]}-set... \n'
    logging.info(log)
    print(log)

    for ann_interval in splits:
        # timestamps as labels (strings)
        # e.g. start_ts = ts1; finish_ts = ts2
        start_ts, finish_ts = list(ann_interval.values())[0].split(',')

        # convert timestamps to ms (strings)
        start_ts = next(
            list(time_stamp.values())[0] for time_stamp in time_stamps
            if list(time_stamp.keys())[0] == start_ts)
        finish_ts = next(
            list(time_stamp.values())[0] for time_stamp in time_stamps
            if list(time_stamp.keys())[0] == finish_ts)

        try:
            # *if such an error is caught, the GERMAN_TO_ENGLISH dic
            # should be updated accordingly
            label = GERMAN_TO_ENGLISH.get(
                list(ann_interval.keys())[0], 'not_found')
            if label == 'not_found':
                log = (f'! Annotation {annotations} is missing the label'
                       f' {list(ann_interval.keys())[0]} \n')
                print(log)
                logging.info(log)
                continue
        except Exception:
            # xml structure is potentially bad
            log = (f'! Key "{list(ann_interval.keys())[0]}" not found.'
                   'XML structure might be faulty')
            print(log)
            logging.info(log)
            logging.info(splits)
            logging.exception('Potentially bad XML structure')
            continue

        # convert to floats
        start_ts = float(start_ts) / 1000
        finish_ts = float(finish_ts) / 1000

        # for each pure clip, i.e. a clip that represents one single annotation
        # generate a bunch of sub-clips
        save_clips(video, label, start_ts, finish_ts, train_test, clip_length,
                   sliding_window, ann_type)


def split_video_to_clips_eval(index, video, annotations, test_split,
                              clip_length, sliding_window, ann_type):
    log = f'\n## Processing video #{index}: {video}...'
    logging.info(log)
    print(log)

    # build xml tree
    tree = ET.parse(annotations)
    root = tree.getroot()

    # get annotations with corresponding timestamp markers
    # [{'walk_straight': 'ts2,ts4'}, ..., {'fall_pos_sit': 'ts55,ts58'}]
    splits = []
    base_ann_ids = ['a' + str(i) for i in range(0, 10)]
    for trier in root.iter('TIER'):
        try:
            eval_annotation = trier[0][0]
        except IndexError:
            # annotations might be unordered and some of them found at the
            # end of the file. So we keep iterating all trier tags to look
            # for annotations
            continue

        if eval_annotation.attrib['ANNOTATION_ID'] in base_ann_ids:
            continue

        # clean the evaluation: remove numbers, columns, etc.
        evaluation = re.sub(r'\d+', '', str(eval_annotation[0].text))
        evaluation = evaluation.replace(':', '').strip()
        evaluation = evaluation.replace('>',
                                        'more than').replace('<', 'less than')
        movement = trier.attrib['TIER_ID'].rsplit('-', 1)[0]
        label = EVALUATION_ANN[movement].get(evaluation, 'not_found')
        if label == 'not_found':
            # *make sure to udate the EVALUATION_ANN dic accordingly
            log = (f'! Annotation {annotations} is missing the label '
                   f'associated with {movement} - {evaluation} \n')
            print(log)
            logging.info(log)
            continue
        elif label == 'not_enough_samples':
            log = (
                f'! Annotation {annotations} does not have enough samples '
                f'for the label associated with {movement} - {evaluation} \n')
            print(log)
            logging.info(log)
            continue
        elif label == 'hard':
            log = (f'! Annotation {annotations} for the label associated with '
                   f'{movement} - {evaluation} is hard for a pure HAR model\n')
            print(log)
            logging.info(log)
            continue
        elif label == 'imbalanced':
            log = (f'! Annotation {annotations} for the label associated with '
                   f'{movement} - {evaluation} is imbalanced\n')
            print(log)
            logging.info(log)
            continue

        splits.append({
            label:
            eval_annotation.attrib['TIME_SLOT_REF1'] + ',' +
            eval_annotation.attrib['TIME_SLOT_REF2']
        })

    # get all timestamps
    # [{'ts1': '3320'}, ..., {'ts58': '235798'}]
    time_stamps = []
    for time_slot in root.iter('TIME_SLOT'):
        time_stamps.append(
            {time_slot.attrib['TIME_SLOT_ID']: time_slot.attrib['TIME_VALUE']})

    np.random.seed()  # make sure to get different results
    train_test = 'videos_train' if np.random.choice(
        [0, 1], p=[1 - test_split, test_split]) == 0 else 'videos_val'
    log = '### Saving clips as {}-set... \n'.format(train_test[7:])
    logging.info(log)
    print(log)

    for ann_interval in splits:
        start_ts, finish_ts = list(ann_interval.values())[0].split(',')
        start_ts = next(
            list(time_stamp.values())[0] for time_stamp in time_stamps
            if list(time_stamp.keys())[0] == start_ts)
        finish_ts = next(
            list(time_stamp.values())[0] for time_stamp in time_stamps
            if list(time_stamp.keys())[0] == finish_ts)

        # convert to floats
        start_ts = float(start_ts) / 1000
        finish_ts = float(finish_ts) / 1000
        save_clips(video,
                   list(ann_interval.keys())[0], start_ts, finish_ts,
                   train_test, clip_length, sliding_window, ann_type)


def extract_clips(vid_items):
    video, index, annotations, args = vid_items
    if args.test_dataset:
        if video not in VALIDATION_SET:
            return
    else:
        if video in VALIDATION_SET:
            return

    annotation = get_video_annotation(video, annotations)
    split_video_to_clips = split_video_to_clips_base\
        if args.ann_type == 'base' else split_video_to_clips_eval

    split_video_to_clips(index, osp.join(args.src_dir, video),
                         osp.join(args.src_dir, annotation), args.test_split,
                         args.clip_length, args.sliding_window, args.ann_type)


def main():
    args = parse_args()
    reset_structure(args.ann_type)
    generate_structure(args.ann_type)
    assert ((args.clip_length > 1) &
            (args.clip_length < 30)), 'Clip length out of bounds (1, 30)'
    assert ((args.sliding_window > 0) &
            (args.sliding_window < 20)), 'Sliding window out of bounds (0, 20)'
    assert (args.clip_length >= args.sliding_window
            ), 'Sliding window must be less than or equal to clip length'

    files = os.listdir(args.src_dir)
    videos = [
        v for v in files if ((v.find('mp4') != -1) | (v.find('MTS') != -1))
    ]
    annotations = [ann for ann in files if osp.splitext(ann)[1] == '.eaf']
    logging.basicConfig(
        filename=get_log_file_name(len(videos), args), level=logging.DEBUG)
    log = (f'\n# Generating dataset for {len(videos)} videos with a clip size'
           f' of {args.clip_length}s, sliding window of'
           f' {args.sliding_window}s and train/test ratio of'
           f' {1-args.test_split}/{args.test_split}...')
    print(log)
    logging.info(log)

    path = TANZ_BASE_PATH if args.ann_type == 'base' else TANZ_EVAL_PATH
    global LABEL_TO_NUMBER
    LABEL_TO_NUMBER = helpers.bast_label_to_number_dict(
        BASE_ANN_PATH if args.ann_type == 'base' else EVAL_ANN_PATH)
    start_time = time.time()

    pool = Pool(args.num_processes)
    try:
        pool.map(
            extract_clips,
            zip(videos, range(0, len(videos)), repeat(annotations),
                repeat(args)))
    finally:
        cleanup()

    if args.test_dataset:
        merge_train_test(path, args.ann_type)
        no_clips = helpers.file_len(
            osp.join(path, f'tanz_{args.ann_type}_test_list_videos.txt'))
    else:
        no_clips = helpers.file_len(
            osp.join(path, 'tanz_train_list_videos.txt')) + helpers.file_len(
                osp.join(path, 'tanz_val_list_videos.txt'))
    log = '# Finished! {} clips created. Execution time: {} minutes'.format(
        no_clips, round((time.time() - start_time) / 60, 1))
    print(log)
    logging.info(log)


if __name__ == '__main__':
    main()
