# Structure-aware and Texture-aware End-to-end Network

This code can be used to reproduce the experiments performed in our paper 'SaTaNet: Structure-aware and Texture-aware End-to-end Network for Cephalometric Landmark Localization on A New Large Dataset'.


<div align="center">
  <div style="display: flex;">
    <img src="https://github.com/21wang12/SaTaNet/assets/38482259/b8a273b9-2b85-4637-a650-29a60909cd59" width="600">
  </div>
</div>
## Requirements

- Python 3 (code has been tested on Python 3.7)
- CUDA and cuDNN (tested with Cuda 11.3)
- Our experiments used a NVIDIA 3090 which has 32GBs of memory. 
- Python packages listed in the requirements.txt including PyTorch 1.10.0

# Set up
To set up environment, please run the following command:
```bash
python3 -m pip install -r requirements.txt
```

# Run train
To train the model, please run the following command:
```bash
CUDA_VISIBLE_DEVICES=XXX python train.py --cfg experiments/cephalometric.yaml --training_images /path/to/training/data/ \
 --annotations /path/to/annotation/data/
```

# Run test
To test the model, please run the following command:
```bash
python temperature_scaling.py --cfg experiments/cephalometric.yaml --fine_tuning_images /data1/wangs/datasets/ISBI2015/CHHeatmaps_process/data/RawImage/Test1Data/ \
 --annotations /path/to/annotation/data/ --pretrained_model /path/to/cephalometric_model.pth
```

