# Self-Paced Self-Training

<br>
<p align="center" style="font-family:courier;font-size:105%;">
  <img src="https://github.com/uvavision/Curriculum-Labeling/blob/main/imgs/framework.png?raw=true" />
  <br>
  System Framework
</p>
<br>

## How to start 
- python=3.7.7
- pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
- pip install -r requirements.txt

## How to run
### Run the commands below to train on CIFAR-10 with WideResNet-28-2 
```
python main.py --doParallel --seed 821 --nesterov --weight-decay 0.0005 --arch WRN28_2 --dataset cifar10 --batch_size 512 --epochs 700 --lr_rampdown_epochs 750 --add_name WRN28_CIFAR10_AUG_MIX_SWA --mixup --swa --num_labeled 400 --num_classes 10 --percentiles_holder 20 --num_valid_samples 500 --classwise_curriculum
```
### Or run script 
```
sh script.sh
```
