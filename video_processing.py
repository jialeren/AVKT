from __future__ import absolute_import, division, print_function
from mouth_cropper import crop_mouth_image

import os
import cv2 as cv
from tqdm import tqdm
from moviepy.editor import AudioFileClip
import torch
import face_alignment
import numpy as np
from skimage import io
import face_recognition
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Video Processing')

    '''是否使用GPU'''
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='cpu or gpu cuda device')

    '''原始视频逐帧提取人脸'''
    parser.add_argument('--video_dir', type=str,
                        default='./test_dataset/', help='')
    parser.add_argument('--faces_dir', type=str,
                        default='./test/faces/', help='')
    parser.add_argument('--gray', type=bool, default=False)

    '''从原始视频中分离音频'''
    parser.add_argument('--audio_dir', type=str, default='./test/audio', help='')

    '''face align and mouth crop'''
    parser.add_argument('--landmarks_dir', type=str, default='./test/landmark', help='the output 68 landmarks path')
    parser.add_argument('--mouth_dir', type=str, default='./test/mouth', help='the output bounding boxes path')
    parser.add_argument('--mouth_size', type=tuple, default=(96, 96), help='[MOUTH] size of cropped mouth ROIs')

    args = parser.parse_args(args=[])
    return args

def extract_faces_from_videos():

    """从视频中提取不同的人脸，以.png格式存放在各自的文件夹"""

    '''忽略图片高度小于0.1*图片宽度的人脸'''
    def face_filter(face_location, threshold=0.1):
        faces1 = []
        for face1 in face_location:
            if abs(face1[0] - face1[2]) > threshold * img.shape[0]:
                faces1.append(face1)
        return faces1

    dst_image_encodings = []
    os.makedirs(args.faces_dir, exist_ok=True)
    file_list = os.listdir(args.video_dir)
    for file in tqdm(file_list):
        video_path = os.path.join(args.video_dir, file)
        write_path = os.path.join(args.faces_dir, file[:-4])
        os.makedirs(write_path, exist_ok=True)  # 按视频名称创建储存被提取人脸图像的文件夹
        frame_reader = cv.VideoCapture(video_path)
        frame_id = 0
        first_frame = True
        while frame_reader.isOpened():
            ret, frame = frame_reader.read()
            if ret:
                img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # OpenCV默认是BGR，转化为RGB读取
                ''' 每个视频的第一帧 '''
                if frame_id == 0:
                    face_locations = face_recognition.face_locations(img,model='cnn')  # cnn
                    faces = face_filter(face_locations)  # 筛选后的脸部
                    for idx, face in enumerate(faces):
                        top = face[0]
                        right = face[1]
                        bottom = face[2]
                        left = face[3]
                        # start = (left, top)
                        # end = (right, bottom)
                        # color = (55, 255, 155)
                        # cv.rectangle(img, start, end, color, 3)
                        '''选一个比脸稍大的框作为剪裁区域'''
                        x = int(0.4 * abs(left - right))
                        y = int(0.4 * abs(top - bottom))
                        cropped = frame[max(0, top - y): min(frame.shape[0], bottom + y),
                                  max(0, left - x): min(frame.shape[1], right + y)]
                        os.makedirs(os.path.join(write_path, str(idx)), exist_ok=True)
                        ''' 将剪裁后的脸保存为.png图片 对每一个人创建一个文件夹(0 1 2)'''
                        ''' /faces/视频文件名/0/0000.png  /faces/视频文件名/0/0001.png ......'''
                        ''' /faces/视频文件名/1/0000.png  /faces/视频文件名/1/0001.png ......'''
                        cv.imwrite(os.path.join(write_path, str(idx), '{:04d}.png'.format(frame_id)),
                                   cropped)  # 将剪裁后的脸保存为.png图片

                        '''将第一帧提取到的图像作为与之后帧的对比'''
                    dst_image_encodings = []
                    person_id = []
                    for direc_id in os.listdir(write_path):
                        dst_image = face_recognition.load_image_file(os.path.join(write_path, direc_id) + "/0000.png")
                        dst_image_encodings.append(face_recognition.face_encodings(dst_image)[0])
                        person_id.append(direc_id)

                        '''第一帧以后的帧,将检测到的每张脸与第一帧的人脸对比'''
                else:
                    face_locations = face_recognition.face_locations(img,model='cnn')
                    faces = face_filter(face_locations)
                    face_encodings = face_recognition.face_encodings(img, faces)
                    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
                        x = int(0.4 * abs(left - right))
                        y = int(0.4 * abs(top - bottom))
                        cropped = frame[max(0, top - y): min(frame.shape[0], bottom + y),
                                  max(0, left - x): min(frame.shape[1], right + y)]
                        '''对于每一张脸，与第一帧重的所有脸计算距离，距离最小的认为是同一个人。如果距离过大，认为不是同一个人，舍去该脸'''
                        face_distances = face_recognition.face_distance(dst_image_encodings, face_encoding)
                        if face_distances[np.argmin(face_distances)] < 0.5:
                            img_name = os.path.join(write_path, str(np.argmin(face_distances)),
                                                    '{:04d}.png'.format(frame_id))
                            # cv.imwrite(img_name, cropped)
                        else:
                            print("距离过大，有新的人脸出现")
                frame_id += 1
            else:
                frame_reader.release()  # 提取完一个视频之后，关闭当前reader才能进入下一个循环
    print(f'{len(file_list)} videos has been extracted to face images!')

# def video2frames():
#     os.makedirs(args.frames_dir, exist_ok=True)
#     filelist = os.listdir(args.video_dir)
#
#     for file in tqdm(filelist):
#         write_path = os.path.join(args.frames_dir, file[:-4])
#         os.makedirs(write_path, exist_ok=True)  # 按视频名称创建储存被提取图像的文件夹
#         frame_reader = cv.VideoCapture(os.path.join(args.video_dir, file))
#         num = 0
#         while frame_reader.isOpened():
#             ret, frame = frame_reader.read()
#             if ret:
#                 if args.gray:
#                     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#                 cv.imwrite(os.path.join(write_path, '{:04d}.png'.format(num)), frame)
#                 num += 1
#             else:
#                 frame_reader.release()  # 提取完一个视频之后，关闭当前reader才能进入下一个循环
#     print(f'{len(filelist)} videos has been extracted to frames!')


def video2audio():
    os.makedirs(args.audio_dir, exist_ok=True)
    file_list = os.listdir(args.video_dir)
    for file in file_list:
        clip = AudioFileClip(os.path.join(args.video_dir, file))
        audio_name = file.replace('mp4', 'wav')
        clip.write_audiofile(os.path.join(args.audio_dir, audio_name))


def face_align():
    os.makedirs(args.landmarks_dir, exist_ok=True)
    dir_list = os.listdir(args.faces_dir)
    for face_dire in dir_list:
        for dire in os.listdir(os.path.join(args.faces_dir,face_dire)):
            fan = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device, flip_input=False)
            preds = fan.get_landmarks_from_directory(os.path.join(args.faces_dir, face_dire, dire),return_bboxes=True)
            for image_file, (landmark, _, box) in preds.items():
                # if not box:
                #     os.makedirs(args.log_path, exist_ok=True)
                #     with open(os.path.join(args.log_path, 'log.txt'), 'a') as logger:
                #         logger.write(os.path.abspath(image_file) + '\n')
                #     continue
                landmark = np.array(landmark)[0]
                npy_file_name = os.path.splitext(os.path.basename(image_file))[0] + '.npy'
                os.makedirs(os.path.join(args.landmarks_dir, face_dire, dire), exist_ok=True)
                image_landmark_path = os.path.join(args.landmarks_dir, face_dire, dire, npy_file_name)
                np.save(image_landmark_path, landmark)


def mouth_crop():
    mean_face = './cropper/20words_mean_face.npy'
    start_idx = 48
    stop_idx = 68
    window_margin = 12
    os.makedirs(args.mouth_dir, exist_ok=True)
    dir_list = os.listdir(args.faces_dir)
    for face_dire in dir_list:
        for dire in os.listdir(os.path.join(args.faces_dir,face_dire)):
            os.makedirs(os.path.join(args.mouth_dir,face_dire,dire),exist_ok=True)
            crop_mouth_image( os.path.join(args.faces_dir, face_dire, dire),
                            os.path.join(args.landmarks_dir, face_dire, dire),
                            os.path.join(args.mouth_dir,face_dire,dire),
                            np.load(mean_face),
                            crop_width=args.mouth_size[0],
                            crop_height=args.mouth_size[1],
                            start_idx=start_idx,
                            stop_idx=stop_idx,
                            window_margin=window_margin )

if __name__ == '__main__':
    args = parse_args()
    extract_faces_from_videos()

