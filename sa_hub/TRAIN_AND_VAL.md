# Training and Validation



### Installation

```shell
cd sa_hub/ultralytics-d8701b42caeb9f7f1de5fd45e7c3f3cf1724ebb6
pip install -e .
cd ..
```

### Preparing Dataset

Set DATA_PATH in sa.yaml

Please refer to https://docs.ultralytics.com/datasets/segment/coco/

NOTE

All category IDs need to be set to 0

### Train

```shell
python train_sa.py
```

### Val

```shell
wget https://huggingface.co/spaces/An-619/FastSAM/resolve/main/weights/FastSAM.pt
python val_sa.py
#You can use the official ULTRALYTICS validation code or save the result into JSON file
```



