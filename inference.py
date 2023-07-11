import argparse
import os
import glob
import json
import tqdm
import time
import imageio
from tqdm import tqdm
from vaik_video_classification_trt_inference.tflite_model import TrtModel

def main(skip_frame, input_saved_model_path, input_classes_path, input_video_dir_path, output_json_dir_path):
    os.makedirs(output_json_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = TrtModel(input_saved_model_path, classes)

    types = ('*.avi', '*.mp4')
    video_path_list = []
    for files in types:
        video_path_list.extend(glob.glob(os.path.join(input_video_dir_path, '*', files), recursive=True))

    total_inference_time = 0
    total_frames_num = 0

    model.inference([frame for frame in imageio.get_reader(video_path_list[0],  'ffmpeg')][::skip_frame])

    for video_path in tqdm(video_path_list):
        # read
        video = imageio.get_reader(video_path,  'ffmpeg')
        frames = [frame for frame in video][::skip_frame]
        total_frames_num += len(frames)
        # inference
        start = time.time()
        output, raw_pred = model.inference(frames)
        end = time.time()
        total_inference_time += (end - start)
        # dump
        output_json_path = os.path.join(output_json_dir_path, os.path.splitext(os.path.basename(video_path))[0]+'.json')

        output_dict = {}
        output_dict['inf'] = output
        output_dict['answer'] = os.path.basename(os.path.dirname(video_path))
        output_dict['video_path'] = video_path
        with open(output_json_path, 'w') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(video_path_list)/total_inference_time}[videos/sec]')
    print(f'{total_frames_num/total_inference_time}[frame/sec]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--skip_frame', type=int, default=16)
    parser.add_argument('--input_saved_model_path', type=str, default='~/model.trt')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-utc101-video-classification-dataset/sub_ucf101_labels.txt')
    parser.add_argument('--input_video_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset/test')
    parser.add_argument('--output_json_dir_path', type=str, default='~/.vaik-video-classification-tflite-experiment/test_inf')
    args = parser.parse_args()

    args.input_saved_model_path = os.path.expanduser(args.input_saved_model_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_video_dir_path = os.path.expanduser(args.input_video_dir_path)
    args.output_json_dir_path = os.path.expanduser(args.output_json_dir_path)

    main(**args.__dict__)