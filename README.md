

# Real-time Segment Anything

[[`Paper`](arxiv/)] [[`Website`](https://huggingface.co/spaces/An-619/FastSAM)] [[`App Link`](#Android-demo)] [[`Model Zoo`](#model-checkpoints)]  [[`BibTeX`](#citing-fastsam)]

 

![FastSAM design](assets/Overview.png?raw=true)

The **Fast Segment Anything Model(FastSAM)** xxx.

- xxx. 
- xxx.
- xxx. 

![FastSAM Speed](assets/head_fig.png?raw=true)


## Installation

Clone the repository locally:

```
git clone git@github.com:facebookresearch/segment-anything.git
```

Create the conda env. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```
conda create -n FastSAM python=3.9
conda activate FastSAM
```

Install the packages:

```
cd FastSAM
pip install -r requirements.txt
```

Install clip:
```
pip install git+https://github.com/openai/CLIP.git
```

## <a name="GettingStarted"></a> Getting Started

First download a [model checkpoint](#model-checkpoints).

Then, you can run the scripts to try the everything mode and three prompt modes.


```
# Everything mode
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
```

```
# text prompt
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg  --text_prompt "the yellow dog"
```

```
# box prompt
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt [570,200,230,400]
```

```
# points prompt
python Inference.py --model_path ./weights/FastSAM.pt --img_path \
        ./images/dogs.jpg  --point_prompt [[520,360],[620,300]] --point_label [1,0]
```

## Demo to try

### Android demo
 
(可以补充FPS)

Please download the [fastsam.apk](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and install it on your android mobile phone. Below is a simple tutorial on how to use it.

放一张使用gif图

(可以补充未来扩展,onnx,视频,point,box)

### Web demo

The [web demo](https://huggingface.co/spaces/An-619/FastSAM) can upload your custom image, and support the similar interface modes like SAM, such as Hover & Click, Box and Everything. 

## <a name="Models"></a>Model Checkpoints

Two model versions of the model are available with different sizes. Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `FastSAM-x`: [YOLOv8x Segment Anything model.](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing)**
- `FastSAM-s`: [YOLOv8s Segment Anything model.](https://drive.google.com/file/d/10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV/view?usp=sharing)

## Results

All result were tested on a single NVIDIA GwDForce RTX 3090

### 1. Inference time
Running Speed under Different Point Prompt Numbers(ms)
| method           | params | 1   | 10  | 100 | E(16x16) | E(32x32*) | E(64x64) |
|------------------|--------|-----|-----|-----|----------|-----------|----------|
| sam-H            | 0.6G   | 446 | 464 | 627 | 852      | 2099      | 6972     |
| sam-B            | 136M   | 110 | 125 | 230 | 432      | 1383      | 5417     |
| FastSAM          | 68M    | 40  |40   | 40  |  40      | 40        | 40       |

### 2. Memory usage

| Dataset    | model            | GPU Memory (MB)        |
|:-----------|:-----------------|:-----------------------|
| COCO 2017  | FastSAM(Yolov8X) | 2608                   |
| COCO 2017  | SAM-H            | 7060                   |
| COCO 2017  | SAM-B            | 4670                   |

### 3. Zero-shot Transfer Experiments

#### Edge Detection
Test on the BSDB500 dataset
|method     |    year|     ODS |     OIS |    AP |  R50 |
|:----------|:-------|:--------|:--------|:------|:-----|
| HED       |    2015| .788    | .808    | .840  | .923 |
| SAM       |    2023| .768    | .786    | .794  | .928 |
| FastSAM   |    2023| .750    | .790    | .793  | .903 |

#### Object Proposals
##### COCO
|method                     | AR10 | AR100 | AR1000 | AUC  |
|---------------------------|------|-------|--------|------|
| SAM-H E64                 | 15.5 | 45.6  | 67.7   | 32.1 |
| SAM-H E32                 | 18.5 | 49.5  | 62.5   | 33.7 |
| SAM-B E32                 | 11.4 | 39.6  | 59.1   | 27.3 |
| FastSAM                   | 15.7 | 47.3  | 63.7   | 32.2 |

##### lvis
bbox AR@1000
| method         | all  | small | med. | large |
|:---------------|:-----|:------|:-----|:------|
| ViTDet-H       | 65.0 | 53.2  | 83.3 | 91.2  |
zero-shot transfer methods
| sam-H E64      | 52.1 | 36.6  | 75.1 | 88.2  |
| sam-H E32      | 50.3 | 33.1  | 76.2 | 89.8  |
| sam-B E32      | 45.0 | 29.3  | 68.7 | 80.6  |
| FastSAM        | 57.1 | 44.3  | 77.1 | 85.3  |

#### Instance Segmentation On COCO 2017

|method         |     AP  |     APS |   APM |  APL |
|:--------------|:--------|:--------|:------|:-----|
| ViTDet-H      | .510    | .320    | .543  | .689 |
| SAM           | .465    | .308    | .510  | .617 |
| FastSAM       | .379    | .239    | .434  | .500 |

### 4.Downstream task

we also do some downstream task that SAM not good at to show the effectiveness of our method.

#### Anomaly Detection

![Anomaly Detection](assets/anomaly.png)

#### Salient Object Detection

![Salient Object Detection](assets/salient.png)

#### Building Extracting

![Building Detection](assets/building.png)

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


## Acknowledgement

- [Segment Anything](https://segment-anything.com/) provides the SA-1B dataset and the base codes.
- [YOLOv8](https://github.com/ultralytics/ultralytics) provides codes and pre-trained models.
- [YOLACT](https://arxiv.org/abs/2112.10003) provides powerful instance segmentation method.
- [Grounded-Segment-Anything](https://huggingface.co/spaces/yizhangliu/Grounded-Segment-Anything) provides a useful web demo template.

 

## Citing FastSAM

If you find this project useful for your research, please consider citing the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anyt是的发送到发送到发送的方式hing},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

<!-- <p align="center">
  <a href="https://star-history.com/#geekyutao/Inpaint-Anything&Date">
    <img src="https://api.star-history.com/svg?repos=geekyutao/Inpaint-Anything&type=Date" alt="Star History Chart">
  </a>
</p> -->
