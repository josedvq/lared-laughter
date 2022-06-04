import os

# paths to raw datasets
raw_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'lared')
raw_audio_path = os.path.join(raw_data_path, 'audio', 'trimmed')
raw_video_path = os.path.join(raw_data_path, 'video', 'concatenated')
audioset_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'audioset')
activitynet_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'activitynet')
kinetics_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'kinetics')
mnm_data_path = os.path.join(os.environ['RAW_DATA_PATH'], 'mnm')

processed_data_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter')

# annotation experiment pilot paths
pilot_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter', 'annotation_pilot')

# annotation experiment paths
annot_exp_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter', 'annotation_experiment')

# dataset paths
datasets_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter', 'datasets')
dataset_path = os.path.join(os.environ['PROC_DATA_PATH'], 'lared_laughter', 'datasets', 'tight')

# models
models_path = '/mnt/e/data/models'