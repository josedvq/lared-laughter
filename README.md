### Code for reproducing the results in the paper: "Differences in annotation of laughter across modalities"

This is a setup for running classification, regression and segmentation tasks on thin slices of multimodel data: video, audio and wearable sensor data, for different label sets (we experimented with labels acquired from different modalities). It also includes code to reproduce the exact HITs used in our human annotation / perception experiments using the Covfee framework.

I used pretrained 3D ResNets for video from the slowfast library, pretraind 2D ResNets for audio (on spectrograms) and a 1D ResNets from the tsai library for acceleration. Features from all networks are fused, before a network head that changes depending on the task.

I implemented caching of the video ResNets features (for each fold, epoch and label type) to be able to reuse them between runs with different input modality combinations (video, audiovisual, audiovisual + accel) and tasks (classification, regression, segementation). This improves total running time by a factor of ~9.

### Data

To be released.

### Running

1. Clone and requirements: 

    ```
    git clone git@github.com:josedvq/lared-laughter.git
    ```

2. Add folder to `PYTHONPATH`. I use Jupyter notebooks with absolute imports in the repo. For these to work correctly, the parent folder of the repository should be in `PYTHONPATH`

    ```
    export PYTHONPATH=$PYTHONPATH:$PWD
    ```

3. Install requirements. I recommend to do it in a virtual environment.
cd lared-laughter
pip install -r requirements.txt

4. You can now play around with the code.
The main tables/runs can be reproduced from the fusion/train.ipynb notebooks.