import os

data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'lared')
audio_path = os.path.join(data_path, 'audio/trimmed')
cloud_data_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter')

laughter_data_path = os.path.join(cloud_data_path, 'laughter_data', 'ml_datasets', 'tight')
audioset_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'audioset')
activitynet_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'activitynet')
kinetics_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'kinetics')
mnm_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'mnm')

video_examples_path = os.path.join(cloud_data_path, 'examples/video')
audio_examples_path = os.path.join(cloud_data_path, 'examples/audio')
av_examples_path = os.path.join(cloud_data_path, 'examples/av')
