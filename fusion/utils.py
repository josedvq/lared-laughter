from matplotlib.pyplot import table
import numpy as np
import pandas as pd
import seaborn as sns



def make_dataframe(res):
    input_mod_map = {
        'accel': 'Acceleration',
        'video': 'Video',
        'audio': 'Audio',
    }

    label_mod_map = {
        'video': 'Video',
        'audio': 'Audio',
        'av': 'Audiovisual'
    }

    t = []
    for input_mod, input_res in res.items():
        index = pd.MultiIndex.from_tuples(
            [('', '', '', 'Input')] +
            [('Label Modality', model_modality, test_modality, value) 
                for model_modality in ['Audio', 'Video', 'Audiovisual']
                for test_modality in ['Audio', 'Video', 'Audiovisual']
                for value in ['mean', 'std']], 
                
            names=["name", "model_mod", "test_mod", "value"])

        row = pd.Series(index=index)

        row[('', '', '', 'Input')] = input_mod_map[input_mod]
        for train_label_mod, train_label_res in input_res.items():

            for test_label_mod, test_label_res in train_label_res.items():
                metrics = np.concatenate([r['metrics'] for r in test_label_res])
                label_mod = label_mod_map[train_label_mod]
                test_mod  = label_mod_map[test_label_mod]
                row[('Label Modality', label_mod, test_mod, 'mean')] = np.mean(metrics)
                row[('Label Modality', label_mod, test_mod, 'std') ] = np.std(metrics)
        
        t.append(row)
    return pd.DataFrame(t)

def make_table(res):
    df = make_dataframe(res)
    for idx, res in df.groupby(level=[0,1,2]):
        df.loc[idx, 'cell'] = f'{res["mean"]:.3f} ({res["std"]:.3f})'

    table_df = df.loc[pd.IndexSlice[:, :, :, 'cell']]
    return table_df

def make_heatmap(res):
    df = make_dataframe(res)
    mean_df = df.loc[pd.IndexSlice[:, :, :, 'mean']]

    return sns.heatmap(mean_df)