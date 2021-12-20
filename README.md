## This code is adapted from https://github.com/srebuffi/residual_adapters

We added our models in ``attention_model.py`` and training script in ``train_new_task_aow.py``.


### Requirements
- Python 2.7
- PyTorch 1.2.0
- Cuda Toolkit 9.2
- Matplotlib 2.1.0
- COCO API

We use the pretrained ImageNet weights from the original repo, which can only be loaded in Python 2.7.

### Launching the code
First download the data with ``download_data.sh /path/to/save/data/``. Please copy ``decathlon_mean_std.pickle`` to the data folder. 

To train our ``channel-wise attention model with direct masking`` and attention dimension ``C/4``:

``python train_new_task_aow.py --dataset [DATASET] --wd 1e-4 --source [PRETRAINED WEIGHTS] --expdir [RESULTS DIR] --batch_size 128 --att_factor 4 ``



To train our ``channel-wise attention with generative masking`` and attention dimension ``C/4``:

``python train_new_task_aow.py --dataset [DATASET] --wd 1e-4 --source [PRETRAINED WEIGHTS] --expdir [RESULTS DIR] --batch_size 128 --att_factor 4 --res``



To train the ``classifier only`` model:

``python train_new_task_aow.py --dataset [DATASET] --wd 1e-4 --source [PRETRAINED WEIGHTS] --expdir [RESULTS DIR] --batch_size 128 --mode original``



### Pretrained networks
We pretrained networks on ImageNet (with reduced resolution):
- a ResNet 26 inspired from the original ResNet from [He,16]: https://drive.google.com/open?id=1y7gz_9KfjY8O4Ue3yHE7SpwA90Ua1mbR


