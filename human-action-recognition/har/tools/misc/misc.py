from rich.console import Console

CONSOLE = Console()

# Methods:
# 1. load_graph - loads a tensorflow graph
#
# 2. merge_train_test() - merges train & test sets from the BAST dataset
# 3. read_pickel() - reads and examines pose-estimation pickle files
# 4. merge_pose() - merges individual pose data into a list of dictionaries
#
# 5. extract_frames_from_video() - extract frames of choice from videos
# 6. merge_images_with_font() - merges several images into one
# 7. gen_id() - generate some random id
# 8. download_youtube() - download single youtube video
# 9. resize() - resize an image
# 10. analyze_avatar() - analyze avatar videos from the BAST analysis
# 11. plotter() - plot pandas dataframe bazed on hue


def plotter(input, hue, out):
    import seaborn as sns
    import pandas as pd

    df = pd.read_csv(input)
    sns.set(rc={'figure.figsize': (16, 13)})
    first = sorted(df.Class.unique())[32:]
    first_df = df[df.Class.isin(first)]
    fig = sns.barplot(x='Class', y='Value', hue=hue, data=first_df)
    fig.set_xticklabels(fig.get_xticklabels(), rotation=30)
    output = fig.get_figure()
    output.savefig(out)


# plotter(
#     'test_acc_per_class.csv',
#     'Accuracy', 'fourth.jpg')


# -----------------------------------------------------------------------------
def analyze_avatar(timestamps, path):
    """`timestamps` json file with timestamps `path` path to files to
    process."""
    import json
    import os
    import pandas as pd
    import seaborn as sns

    ts_content = json.load(open(timestamps, 'r'))
    result = {}
    for key in ['water', 'fire', 'air', 'earth']:
        result[key] = {}
        for task in [
                'tiptoe', 'walk', 'run', 'rotate', 'contract-expand',
                'swing-upper-body', 'jump', 'stamp', 'fall'
        ]:
            result[key][task] = 0

    for item in os.listdir(path):
        if (not item.endswith('.json')) | (item not in [
                video['video'] for video in ts_content
        ]):
            continue
        CONSOLE.print(item)
        video_content = json.load(open(os.path.join(path, item), 'r'))

        for video in ts_content:
            if video['video'] == item:
                for action in ['water', 'fire', 'air', 'earth']:
                    CONSOLE.print(action)
                    start = round(video[action][0] / 10)
                    finish = round(video[action][1] / 10)
                    for i in range(start, finish):
                        key = f"('ts{i:02}', 'ts{i+1:02}')"
                        CONSOLE.print(key)
                        performed_action = video_content.get(key, 'Not Found')
                        top1 = performed_action.split(',')[0]

                        if top1 != 'Not Found':
                            result[action][top1] += 1
                            top2 = performed_action.split(',')[1].split(' ')[1]
                            result[action][top2] += 1
                            top3 = performed_action.split(',')[2].split(' ')[1]
                            result[action][top3] += 1

    # write as json
    result_json = json.dumps(result, indent=4)
    f = open('avatar_analysis.json', 'w')
    print(result_json, file=f)
    f.close()

    # plot
    df = pd.DataFrame(result)
    sns.set(rc={'figure.figsize': (13, 13)})
    # fig = sns.barplot(data=df) #, hue=df.columns.to_numpy())
    fig = df.plot(kind='bar', color=['blue', 'red', 'white', 'brown'])
    fig.set_xticklabels(fig.get_xticklabels(), rotation=20)
    fig.axes.set_title('Second Task Analysis', fontsize=30)
    fig.set_xlabel('Task', fontsize=30)
    fig.set_ylabel('', fontsize=10)
    output = fig.get_figure()
    output.savefig('avatar_analysis.jpg')


# analyze_avatar(
#   'scripts-local/avatar_videos_segments.json',
#   'minio-transfer/read/avatar_vids')


# -----------------------------------------------------------------------------
def gen_id(size=8):
    """Generate a random id."""
    import string
    import random
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


# -----------------------------------------------------------------------------
def resize(img, shape=(480, 480)):
    from moviepy.editor import ImageClip
    ImageClip(img).resize(shape).save_frame(f'{gen_id(4)}.jpg')


# resize('playing tenis.jpg', (320, 320))


# -----------------------------------------------------------------------------
def download_youtube(link):
    from pytube import YouTube
    YouTube(link).streams.first().download()


# download_youtube('https://www.youtube.com/watch?v=notLDzBJ2mg&t=66s')


# -----------------------------------------------------------------------------
def merge_images_with_font(*args,
                           cols=2,
                           label_rgb=(204, 204, 0),
                           label_pos=(50, 0),
                           font_size=40):
    """Merge two or more images of same size into one based on #cols using
    Pillow True Type Fonts: https://ttfonts.net."""
    from PIL import Image, ImageOps, ImageFont, ImageDraw
    import math

    fnt = ImageFont.truetype(
        'scripts-local/fonts/07558_CenturyGothic.ttf', size=font_size)

    images = []
    for arg in args:
        # open images & add border
        img = ImageOps.expand(Image.open(arg), border=3, fill='blue')
        # add the label to the images
        img = ImageDraw.Draw(img)
        if label_pos is not None:
            img.text(
                label_pos,
                arg.split('.')[0],
                font=fnt,
                fill=label_rgb,
                align='center')
        images.append(img._image)

    size = images[0].size  # (320, 240)
    rows = math.ceil(len(images) / cols)

    # create the new image based on #cols & #rows
    result = Image.new('RGB', (size[0] * cols, size[1] * rows), 'white')

    # add the images to the new image
    c = 0
    for i in range(rows):
        for j in range(cols):
            result.paste(images[c], (j * size[0], i * size[1]))
            c += 1
            if c == len(images):
                break

    result.save('merged_result.jpg')


merge_images_with_font(
    '22_6K6T.jpg',
    '23_TYP6.jpg',
    '25_5SEW.jpg',
    '24_OHPI.jpg',
    '27_PSRB.jpg',
    '28_KQUL.jpg',
    cols=3,
    label_rgb=(128, 128, 128),
    font_size=40)

# -----------------------------------------------------------------------------


def extract_frames_from_video(video_path, pos=0, dims=None):
    """Extract frames at a given position of a video using moviepy `dims` is a
    tuple containing width and height."""
    from moviepy.editor import VideoFileClip
    from moviepy.video.fx.resize import resize

    with VideoFileClip(video_path) as video:
        print(f'Video FPS: {video.fps}')
        frame = video.to_ImageClip(pos)
    if dims is not None:
        frame = resize(frame, dims)

    frame.save_frame(f'{pos}_{gen_id(4)}.jpg')


# for i in np.arange(21, 30, 1):
# extract_frames_from_video('current.mp4', i)

# -----------------------------------------------------------------------------


def merge_pose(path, split):
    """Given the pose estimation of single videos stored as dictionaries in.

    .pkl format, merge them together and form a list of dictionaries.

    Args:
        path ([string]): path to the pose estimation for individual clips
        split ([string]): train, val, test
    """
    import os
    import os.path as osp
    import pickle
    result = []
    for ann in os.listdir(path):
        if ann.endswith('.pkl'):
            with open(osp.join(path, ann), 'rb') as f:
                annotations = pickle.load(f)
        result.append(annotations)
    with open(f'bast_{split}.pkl', 'wb') as out:
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)


# merge_pose('minio-transfer/read/pkl', 'train')
# -----------------------------------------------------------------------------


def read_pickel(path):
    import pickle
    with open(path, 'rb') as f:
        annotations = pickle.load(f)

    print(f'Type: {type(annotations)}')
    print(f'Length: {len(annotations)}')
    if type(annotations) is list:
        print(f'Keys: {annotations[0].keys()}')
        print(annotations[0])
    else:
        f_no = len(annotations['keypoint'][0])
        pos = int(f_no / 2)
        print(f'Keys: {annotations.keys()}')
        print(annotations['keypoint'][0]
              [pos])  # keypoint[0] because there is only one person
        print(annotations['keypoint_score'][0][pos])
        print('\n\n\n')
        print(annotations)


# read_pickel('minio-transfer/read/posec3d/gym_val.pkl')
# read_pickel('minio-transfer/read/bast_train.pkl')

# -----------------------------------------------------------------------------


# * Merge train & test set FROM BAST dataset
def merge_train_test(path):
    import os
    import shutil
    import os.path as osp

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
            line.replace('videos_val', 'videos_train')
            file.write(line)

    # restructure
    shutil.rmtree(val_path)
    shutil.rmtree(osp.join(path, 'annotations'))
    os.unlink(osp.join(path, 'tanz_val_list_videos.txt'))
    os.rename(train_path, osp.join(path, 'clips_eval'))
    os.rename(
        osp.join(path, 'tanz_train_list_videos.txt'),
        osp.join(path, 'tanz_test_list_videos.txt'))

    print('Merged videos_val with videos_train')


# merge_train_test('minio-transfer/read/tanz')

# -----------------------------------------------------------------------------


# * loading TensorFlow Graph given *pb or *pbmm files
def load_graph(frozen_graph_filename):
    import tensorflow as tf
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import
    # a graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None)
    return graph


def load_graph2(graph_def_pb_file):
    import tessorflow as tf
    with tf.gfile.GFile(graph_def_pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as _:
            tf.import_graph_def(graph_def, name='')


# -----------------------------------------------------------------------------


def long_video_demo():
    import subprocess
    import os
    from tqdm import tqdm
    out_path = '/mnt/data_transfer/write/avatar_vids/'
    in_path = '/mnt/data_transfer/read/to_process_test/avatar_vid/'
    existing = os.listdir(out_path)

    for vid in tqdm(os.listdir(in_path)):
        out = vid.split('.')[0] + '.json'
        if out in existing:
            continue
        print(f'Processing {vid}...')
        subargs = [
            'python',
            'human-action-recognition/har/tools/long_video_demo_clips.py',
            os.path.join(in_path, vid),
            ('configs/skeleton/posec3d/'
             'slowonly_r50_u48_240e_ntu120-pr_keypoint_bast.py'),
            ('/mnt/data_transfer_tuning/write/work_dir/8/'
             '56f6783167af4c75835f2021a30bd136/artifacts/'
             'best_top1_acc_epoch_425.pth'),
            os.path.join(out_path,
                         out), '--num-processes', '25', '--num-gpus', '3'
        ]
        subprocess.run(subargs)


# -----------------------------------------------------------------------------


def demo_posec3d(path):
    import subprocess
    import os
    from tqdm import tqdm

    script_path = 'demo/demo_posec3d.py'
    config = ('configs/skeleton/posec3d/'
              'slowonly_r50_u48_240e_ntu120-pr_keypoint_bast.py')
    checkpoint = ('/mnt/data_transfer_tuning/write/work_dir/10/'
                  '4f6aa64c148544198e26bbaf50da2100/artifacts/'
                  'best_top1_acc_epoch_225.pth')
    ann = ('human-action-recognition/har/annotations/BAST/eval/'
           'tanz_annotations_42.txt')

    for clip in tqdm(os.listdir(path)):
        subargs = [
            'python',
            script_path,
            os.path.join(path, clip),
            os.path.join(path, clip),  # overwrite original clip
            '--config',
            config,
            '--checkpoint',
            checkpoint,
            '--label-map',
            ann,  # class annotations
            '--device',
            'cuda:0'
        ]
        subprocess.run(subargs)
