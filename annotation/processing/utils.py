import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from natsort import natsorted

class CovfeeParser:

    def __init__(self, laughter_examples, speech_examples, calibration_examples, hits):
        self.laughter_examples = laughter_examples
        self.speech_examples = speech_examples
        self.calibration_examples = calibration_examples
        
        self.hits = hits


    def _get_hit_for_results(self, results):
        target = [t for t in results['tasks'].values() if t['gt_laughter']][0]
        target_hash = target['hash']
        target_condition = target['condition']

        def find_in_task(t):
            if t['type'] == 'shuffle':
                return any([find_in_task(t) for tg in t['tasks'] for t in tg])
            elif t['name'] == 'Recognition':
                parts = t['id'].split('_')
                hash = parts[1]
                condition = parts[2]

                return ((hash == target_hash) & (condition == target_condition))

        def find_in_hit(hit):
            return any([find_in_task(t) for t in hit['tasks']])

        found = [find_in_hit(hit) for hit in self.hits['hits']]
        assert np.sum(found) == 1

        hit_idx = np.where(found)[0][0]
        return self.hits['hits'][hit_idx]


    def parse_hit(self, hit_path: Path, data_col='data0', version=1):
        instance_id = str(version) + '_' + os.path.basename(hit_path)
        results_dict = {}
        continuous_annotations = {}

        num_segments = 0

        # process the hit info
        hitfiles = list(hit_path.glob('*.json'))
        hitfiles = [str(e) for e in hitfiles]
        hitfiles = natsorted(hitfiles)
        start_datetime = datetime.fromisoformat(json.load(open(hitfiles[0]))['submitted_at'])
        end_datetime = datetime.fromisoformat(json.load(open(hitfiles[-1]))['submitted_at'])
        duration = (end_datetime - start_datetime).seconds / 60

        assert 'Feedback' in str(hitfiles[-1]), f'Feedback not in {str(hitfiles[-1])}'
        feedback = json.load(open(hitfiles[-1]))

        

        # read the JSON (rating) files
        for i_file, json_file in enumerate(hit_path.glob('*.json')):
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
                'rect': example['rect'],
                'instance_id': instance_id, 
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
                'confidence': json_res.get('confidence', None),
                'intensity': json_res.get('intensity', None)
            }
            num_segments += 1

        # read the CSV (continuous) files
        csv_files = [f for f in hit_path.glob('*.csv')]
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
                # print(f'media_time  all zeroes for {csv_file}')
                continue
            pressed_key = cont_data[data_col].any()

            cd_nonzero = cont_data[cont_data['media_time'] != 0].reset_index()
            cont_signal = cd_nonzero[data_col].to_numpy()
            
            onset_indices  = (np.where(np.diff(cont_signal) ==  1)[0])
            offset_indices = (np.where(np.diff(cont_signal) == -1)[0])

            onset_times = cd_nonzero['media_time'][onset_indices].to_list()
            offset_times= cd_nonzero['media_time'][offset_indices].to_list()

            if cont_signal[0] == 1:
                onset_times.insert(0, 0)
            if cont_signal[-1] == 1:
                offset_times.append(cd_nonzero['media_time'][len(cont_signal)-1])

            assert len(onset_times) == len(offset_times), str(cont_signal)
            for i, j in zip(onset_times, offset_times):
                assert j > i, str((i,j))

            # print(cd_nonzero.to_string())
            # print((onset_times, cont_data[cont_data[data_col] == 1].iloc[0]['media_time'] if pressed_key else None))

            results_dict[example_hash] = {
                **results_dict[example_hash],
                'has_continuous': True,
                'attempt': attempt,
                'pressed_key': pressed_key,
                'onset_times': np.array(onset_times),
                'offset_times': np.array(offset_times),
                'onset': onset_times[0] if len(onset_times) > 0 else None,
                'offset': offset_times[-1] if len(offset_times) > 0 else None
            }

            continuous_annotations[example_hash] = cont_data[['media_time', data_col]]
            continuous_annotations[example_hash] = continuous_annotations[example_hash].rename(columns={data_col: 'data'})

        res = {
            'hit':instance_id ,
            'version': version,
            'tasks': results_dict, 
            'continuous': continuous_annotations,
            'duration': duration,
            'rating': feedback['response']['rating'],
            'feedback': feedback['response']['feedback']
        }
        hit_data = self._get_hit_for_results(res)
        hit_name = hit_data['name']
        hit_group = hit_name[1]
        hit_num   = hit_name[-1]
        for t in res['tasks'].values():
            t['G'] = hit_group
            t['N'] = hit_num

        return {
            'hit_name': hit_name,
            'hit_group': hit_group,
            'hit_num': hit_num,
            **res
        }
        

    def parse_v1(self, results_path, data_col='data0', version=1):
        all_results = {}

        p = Path(results_path)
        for i_dir, hit_path in enumerate(p.iterdir()):
            instance_id = str(version) + '_' + os.path.basename(hit_path)
            all_results[instance_id] = self.parse_hit(hit_path, data_col, version)
            # print(f'HIT {instance_id}, segments: {num_segments}')

        return all_results

    def parse_v2(self, results_path):
        return self.parse_v1(results_path, data_col='data1', version='2')

def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {i: 0 for i in range(1,8)}
    for i in seq:
        hist[i] = hist[i] + 1
    return hist        

def get_hit_stats(hit):
    ''' Filter hits that are completely invalid and return them separately
    '''
    valid_hits = {}
    invalid_hits = {}

    tasks = [t for t in hit['tasks'].values() if not t['calibration']]
    num_tasks = len(tasks)
    num_with_continuous = np.sum([t.get('has_continuous', False) for t in tasks])

    num_pressed_key = np.sum([t.get('pressed_key', False) for t in tasks
                            if t.get('has_continuous', False)])

    num_is_laughter = np.sum([t.get('is_laughter', False) for t in tasks])
    num_intensity_not_none = np.sum([t['intensity'] is not None for t in tasks])
    intensities = count_elements([t['intensity'] for t in tasks if (t['is_laughter'] and t['intensity'] is not None)])
    confidences = count_elements([t['confidence'] for t in tasks if t['confidence'] is not None])

    hit_group = hit["hit_name"][1]
    hit_num   = hit["hit_name"][-1]

    return {
        'id': f'{hit["hit"][:8]}',
        # 'hit_name': hit["hit_name"],
        'hit_group': hit_group,
        'hit_num': hit_num,
        'cont': f'{num_with_continuous}/{num_tasks}',
        'pressed': f'{num_pressed_key}/{num_with_continuous}',
        'is_laughter': f'{num_is_laughter}/{num_tasks}',
        'intensity_not_none': f'{num_intensity_not_none}/{num_tasks}',
        'intensities': '-'.join([f'{c:02d}' for c in intensities.values()]),
        'confidences': '-'.join([f'{c:02d}' for c in confidences.values()]),
        'duration': hit['duration'],
        'rating': hit['rating']
    }

def interp_30fps(df, example_len, data_column='data'):
    sel = df['media_time'] != 0
    sel[0] = True
    df = df[sel]

    if len(df) == 0:
        raise Exception()

    f = interp1d(df['media_time'].to_numpy(), df[data_column].to_numpy(),
        kind='nearest',
        fill_value='extrapolate')
    x = np.arange(0, example_len, 1/30)

    return f(x)