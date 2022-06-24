import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from natsort import natsorted

class CovfeeParser:

    def __init__(self, laughter_examples, speech_examples, calibration_examples):
        self.laughter_examples = laughter_examples
        self.speech_examples = speech_examples
        self.calibration_examples = calibration_examples

    def parse_v1(self, results_path):
        all_results = []
        hit_info = []

        p = Path(results_path)
        for i_dir, dir in enumerate(p.iterdir()):
            hit_id = os.path.basename(dir)
            results_dict = {}
            continuous_annotations = {}

            num_segments = 0

            # process the hit info
            hitfiles = list(dir.glob('*.json'))
            hitfiles = [str(e) for e in hitfiles]
            hitfiles = natsorted(hitfiles)
            start_datetime = datetime.fromisoformat(json.load(open(hitfiles[0]))['submitted_at'])
            end_datetime = datetime.fromisoformat(json.load(open(hitfiles[-1]))['submitted_at'])
            duration = (end_datetime - start_datetime).seconds / 60

            assert 'Feedback' in str(hitfiles[-1]), f'Feedback not in {str(hitfiles[-1])}'
            feedback = json.load(open(hitfiles[-1]))

            hit_info.append({
                'duration': duration,
                'rating': feedback['response']['rating'],
                'feedback': feedback['response']['feedback']
            })

            # read the JSON (rating) files
            for i_file, json_file in enumerate(dir.glob('*.json')):
                # get example index
                
                fname = os.path.basename(json_file).split('.')[0]
                if 'example' in fname or 'rating' not in fname:
                    continue

                # {index}_{recognition|rating}_{hash}_{condition}_{block}_{1 if calibration else 0}'
                # 14_rating_47d49b6e15d4befe8e7a9508068046a7d5478da30f1cc77074951c94cdef9439_video_1_0_0
                parts = fname.split('_')

                index_in_hit = int(parts[0])
                example_hash = parts[2]
                condition = parts[3]
                block = int(parts[4])
                calibration = (parts[5] == '1')

                # read the json response
                json_res = json.load(open(json_file))['response']
                if json_res is None:
                    json_res = {}
                    print(f'json_res is None for file {json_file}')

                # get the ground truth
                if example_hash in self.laughter_examples:
                    example = self.laughter_examples[example_hash]
                    gt_laughter = True
                elif example_hash in self.calibration_examples:
                    example = self.calibration_examples[example_hash]
                    gt_laughter = True
                elif example_hash in self.speech_examples:
                    example = self.speech_examples[example_hash]
                    gt_laughter = False
                else:
                    raise Exception(f'example hash {example_hash} not found anywhere for {i_dir}, {i_file}, {fname}')

                results_dict[example_hash] = {
                    'person': example['pid'], 
                    'cam': example['cam'],
                    'hit_id': hit_id, 
                    'condition': condition,
                    'calibration': calibration,
                    'hash': example_hash,
                    'ini_time': example['ini_time'],
                    'end_time': example['end_time'],
                    '_ini_time': example['_ini_time'],
                    '_end_time': example['_end_time'],
                    'gt_onset': example['onset_time'],
                    'gt_offset': example['offset_time'],
                    'gt_laughter': gt_laughter,
                    'is_laughter': json_res.get('laughter', True),
                    'confidence': json_res.get('confidence', 4),
                    'intensity': json_res.get('intensity', 4)
                }
                num_segments += 1

            # read the CSV (continuous) files

            csv_files = [f for f in dir.glob('*.csv')]
            for csv_file in sorted(csv_files):
                fname = os.path.basename(csv_file).split('.')[0]
                # read only 'recognition' files for the non-example files
                if 'example' in fname or 'recognition' not in fname:
                    continue
                parts = fname.split('_')
                attempt = int(parts[-1])
                example_hash = parts[2]

                cont_data = pd.read_csv(csv_file, index_col=0, header=0)

                if (cont_data['media_time'] == 0).all():
                    # skip invalid continuous annotations
                    print(f'media_time  all zeroes for {csv_file}')
                    continue
                pressed_key = cont_data['data0'].any()

                results_dict[example_hash] = {
                    **results_dict[example_hash],
                    'has_continuous': True,
                    'attempt': attempt,
                    'pressed_key': pressed_key,
                    'onset': cont_data[cont_data['data0'] == 1].iloc[0]['media_time'] if pressed_key else None,
                    'offset': cont_data[cont_data['data0'] == 1].iloc[-1]['media_time'] if pressed_key else None
                }

                continuous_annotations[example_hash] = cont_data[['media_time', 'data0']]

            all_results.append({'hit':hit_id ,'processed': results_dict, 'continuous': continuous_annotations})
            print(f'HIT {hit_id}, segments: {num_segments}')

        return all_results, hit_info

def interp_30fps(df, example_len):
    sel = df['media_time'] != 0
    sel[0] = True
    df = df[sel]

    if len(df) == 0:
        raise Exception()

    f = interp1d(df['media_time'].to_numpy(), df['data0'].to_numpy(),
        kind='nearest',
        fill_value='extrapolate')
    x = np.arange(0, example_len, 1/30)

    return f(x)