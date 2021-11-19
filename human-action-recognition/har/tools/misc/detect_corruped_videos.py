import os
import re
import time

import cv2
import decord as de
import ffmpeg

PATH_TO_CLIPS = '../../minio-transfer/write/tanz/'

if __name__ == '__main__':
    context = de.gpu(0)
    frames = {}
    start = time.time()
    logs = []

    for tt in ['videos_train', 'videos_val']:
        to_tt = os.path.join(PATH_TO_CLIPS, tt)
        for label in os.listdir(to_tt):
            to_tt_label = os.path.join(to_tt, label)

            for clip in os.listdir(to_tt_label):
                print('\n#Examining {} '.format(to_tt_label))
                to_clip = os.path.join(to_tt_label, clip)

                # (1) read frames with decord
                # If they aren't read then there should be some error
                vr = de.VideoReader(to_clip)
                frames = {**frames, **{to_clip: len(vr)}}

                # (2) check if video is corrupted with ffmpeg
                # https://github.com/kkroening/ffmpeg-python/issues/282
                try:
                    (ffmpeg.input(to_clip).output('null', f='null').run())
                except ffmpeg._run.Error:
                    logs.append('!!!!!     CORRUPT VIDEO: {}'.format(to_clip))
                else:
                    pass

                # (3)
                # https://superuser.com/questions/100288/how-can-i-check-the-integrity-of-a-video-file-avi-mpeg-mp4
                t = os.popen('ffmpeg -v error -i "%s" -f null - 2>error.log' %
                             to_clip).read()
                #
                # (4)
                # https://stackoverflow.com/questions/58815980/how-can-i-tell-if-a-video-file-is-corrupted-ffmpeg
                t2 = os.popen('ffmpeg -i "%s" -c copy -f null - 2>&1' %
                              to_clip).read()
                t3 = os.popen('ffmpeg -i "%s" -f null - 2>&1' % to_clip).read()
                t = re.sub(r'frame=.+?\r', '', t)
                t = re.sub(r'\[(.+?) @ 0x.+?\]', '[\\1]', t)
                if len(t) > 1:
                    logs.append('#T' + t)
                if len(t2) > 1:
                    logs.append('#T2' + t2)
                if len(t3) > 1:
                    logs.append('#T3' + t3)

                # (#5) Using openCV
                # https://stackoverflow.com/questions/49750691/python-check-for-corrupted-video-file-catch-opencv-error
                try:
                    vid = cv2.VideoCapture(to_clip)
                    if not vid.isOpened():
                        logs.append(to_clip)
                except cv2.error as e:
                    print(e)
                    logs.append(to_clip)
                except Exception as e:
                    print(e)
                    logs.append(to_clip)

    print('\n\n# Finished: {}min'.format(round((time.time() - start) / 60, 2)))
    # mAx, mIn = max(frames.values()), min(frames.values())
    # for k, v in frames.items():
    #     if v == mAx:
    #         print('Clip {} has most frames: {}'.format(k, v))
    #     if v == mIn:
    #         print('Clip {} the least frames: {}'.format(k, v))
    print('\n\n\n')
    print(len(logs))
    with open('error_logs.txt', 'w') as out:
        for log in logs:
            out.write(log)
    print('Saved results')
