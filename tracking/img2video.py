import cv2
import numpy as np
import glob
import os

video_save_path = '/home/cx/cx2/vis/videos'
frames_paths = [
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/tank-14',
    '/home/cx/cx2/vis/goten_l384_2t_10_80/got10k/GOT-10k_Test_000002',
    '/home/cx/cx2/vis/goten_l384_2t_10_80/got10k/GOT-10k_Test_000036',
    '/home/cx/cx2/vis/goten_l384_2t_10_80/got10k/GOT-10k_Test_000041',
    '/home/cx/cx2/vis/goten_l384_2t_10_80/got10k/GOT-10k_Test_000049',
    '/home/cx/cx2/vis/goten_l384_2t_10_80/got10k/GOT-10k_Test_000077',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/bus-5',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/car-2',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/drone-7',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/drone-2',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/drone-15',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/kite-15',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/licenseplate-12',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/motorcycle-18',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/racing-10',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/racing-15',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/racing-16',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/sheep-5',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/tank-16',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/train-20',
    '/home/cx/cx2/vis/goten_l384_2t_5_80/lasot/truck-16',
    '/home/cx/cx2/vis/goten_l224/lasot_extension_subset/lantern-3',
    '/home/cx/cx2/vis/goten_l224/lasot_extension_subset/lantern-5',
    '/home/cx/cx2/vis/goten_l224/lasot_extension_subset/wingsuit-9'
]

for frames_path in frames_paths:
    video_name = frames_path.split('/')[-1]+'.avi'
    path_img = os.path.join(frames_path, '*.jpg')
    path_save = os.path.join(video_save_path, video_name)
    img_array = []
    filenames = glob.glob(path_img)
    filenames.sort()
    for filename in filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(path_save,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()