import os

data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'lared')
audio_path = os.path.join(data_path, 'audio/trimmed')
cloud_data_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter')

audioset_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'audioset')
activitynet_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'activitynet')
kinetics_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'kinetics')

video_examples_path = os.path.join(cloud_data_path, 'examples/video')
audio_examples_path = os.path.join(cloud_data_path, 'examples/audio')
av_examples_path = os.path.join(cloud_data_path, 'examples/av')
