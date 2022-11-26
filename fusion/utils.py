from matplotlib.pyplot import table
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _get_index(values):
    index = pd.MultiIndex.from_tuples(
            [(model_modality, test_modality, value) 
                for model_modality in ['Audio', 'Video', 'Audiovisual']
                for test_modality in ['Audio', 'Video', 'Audiovisual']
                for value in values], 
                
            names=["Model Modality", "Label Modality", "Quantity"])

    return index

def make_dataframe(res):
    input_mod_map = {
        'video': 'Video',
        'audio': 'Audio',
        'audio-video': 'Audio+Video',
        'accel': 'Acceleration',
        'video-accel': 'Accel+Video',
        'audio-video-accel': 'Accel+Audio+Video'
    }

    label_mod_map = {
        'video': 'Video',
        'audio': 'Audio',
        'av': 'Audiovisual'
    }

    t = {}
    for input_mod in input_mod_map.keys():
        input_res = res[input_mod]
        row = pd.Series(index=_get_index(values=['mean', 'std']), dtype=np.float32)

        for train_label_mod, train_label_res in input_res.items():
            for test_label_mod, test_label_res in train_label_res.items():
                metrics = np.concatenate([r['metrics'] for r in test_label_res])
                label_mod = label_mod_map[train_label_mod]
                test_mod  = label_mod_map[test_label_mod]
                row[(label_mod, test_mod, 'mean')] = np.mean(metrics)
                row[(label_mod, test_mod, 'std') ] = np.std(metrics)
        
        t[input_mod_map[input_mod]] = row
    return pd.DataFrame(t).transpose()

def make_table(res, fmt='%.3f'):
    df = make_dataframe(res)
    for row_idx, row in df.iterrows():
        res = row.groupby(level=[0,1])
        for col_idx, res in res:
            df.loc[row_idx, (*col_idx, 'cell')] = f'{fmt % res[0]} ({fmt % res[1]})'

    # table_df = df.loc[:, pd.IndexSlice[:, :, 'cell']]
    return pd.DataFrame(df.loc[:, pd.IndexSlice[:, :, 'cell']], columns=_get_index(values=['cell']))

def make_heatmap(res, reverse=False, fmt='%.3f'):
    df = make_dataframe(res)
    mean_df = df.loc[:, pd.IndexSlice[:, :, 'mean']].transpose()
    mean_df.index = [f'{row[0]} > {row[1]}' for row in _get_index(values=['mean'])]
    # mean_df.index = [' '.join(col).strip() for col in df.columns.index]

    annot = make_table(res, fmt=fmt).transpose()

    fig, ax = plt.subplots(1,1, figsize=(12,4))

    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')

    # return sns.heatmap(mean_df, ax=ax, annot=True, fmt='.3f')
    cmap = sns.cm.rocket_r if reverse else sns.cm.rocket
    sns.heatmap(mean_df, ax=ax, annot=annot, fmt='s', cmap=cmap)
    return fig