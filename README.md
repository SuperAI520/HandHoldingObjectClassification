# HandHoldingObjectClassification

## Introduction
This repository is implemented to classify whether hand is holding object or not.

Various model architectures are used in training a classficiation model to comparing the classification results.

|Architecture|
|------------|
|MobileFace  |
|Res50       |
|Res50_IR    |
|SERes50_IR  |
|Res100_IR   |
|SERes100_IR |
|Attention_56|
|Attention_92|

As a margin layer, Softmax is used.

Output category is [**Holding Nothing, Holding Object**].

Dataset includes 435 training images, 18 test images.
|Holding Nothing|Holding Object|
|---------------|--------------|
|![holdnothing_72](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/16933fbf-118a-437d-ae94-39b0b6222aa9)|![holdobject_206](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/a7d9c74b-aeb4-4285-a8fa-703ffe66157f)|
|![holdnothing_71](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/143b0856-d810-426e-abbf-de06b7278c8b)|![holdobject_203](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/754dc957-28d2-4d3c-8091-743a63277d4f)|
|![holdnothing_70](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/d5a31691-c6d1-4e58-8c32-32a5404f1e80)|![holdobject_201](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/2be464e5-8e96-4a30-98fa-9883f0bd899a)|
|![holdnothing_69](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/72c2695c-0abf-4038-b661-301e490d5b17)|![holdobject_186](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/522e55ef-6e52-41bb-951b-e14070ac3e3b)|

## How to Use
In order to train your own custom dataset with this implementation, place your dataset folder at the root directory. Make sure that your dataset is split into two subfolder. train and test where the former contains your training dataset and the latter contains your validation set. Refer to the folder named **datasets** in the repo as an example.

```
python train.py --train_root [YOUR TRAIN DATASET FOLDER NAME] --train_file_list [YOUR TRAIN DATASET FILE LIST] --test_root [YOUR TEST DATASET FOLDER NAME] --test_file_list [YOUR TEST DATASET FILE LIST] --backbone [MODEL BACKBONE] --gpus [NUMBER OF GPU]
```

```
python test.py --source [SOURCE IMAGE PATH] --backbone [MODEL BACKBONE] --net_path [MODEL WEIGHT PATH] --margin_path [MARGIN PATH] --gpus [NUMBER OF GPU]
```

## Performance

### MobileFace
![MobileFace](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/0ae4bfff-c422-4346-8efa-60b6c9ea3374)

### Resnet50
![Res50](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/dd507868-9e7b-4a6f-a7bb-bb4103d2889e)

### Resnet50_IR
![Res50_IR](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/3a16a369-4823-4320-9041-c5c2fba0227c)

### SEResnet50_IR
![SERes50_IR](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/507194c3-8996-41ea-a69b-8cd311c1e13c)

### Resnet100_IR
![Res100_IR](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/b948e80e-9f81-4c19-b70e-06fd14fd15ee)

### SEResnet100_IR
![SERes100_IR](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/fe230330-f35b-4a30-a86e-2c525d5354d0)

### Attention_56
![Attention56](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/bb03e540-c557-4289-ac06-6db0aa95ef23)

### Attention_92
![Attention92](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/5be2b857-8eee-49dd-be62-a3708f135ac9)

![compare](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/99a268de-f467-42ea-8b90-186ec4cbf41a)


## Model weights
To test the repository, download pretrained weights from [here](https://drive.google.com/file/d/1ULM9xIQ3HBlmm0hWGtyNtH5KywVB8pUw/view?usp=sharing)

## Visualization
Here are the classification results for MobileFace net.

### Holding Object
|  |  |
|--|--|
|![holdobject_6](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/51865494-6c90-4f48-b8ef-9526f7db1ad9)|![holdobject_5](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/8a362577-3ae1-40b8-808b-05e2c1f5bd60)|
|![holdobject_4](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/4c89be97-8fa6-4526-8207-3f5b872bd674)|![holdobject_3](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/274b2fb5-2c31-45bd-945b-6fd4c7c6253e)|

### Holding Nothing
|  |  |
|--|--|
|![holdnothing_5](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/77a0102f-19b5-471c-8443-2dcfd40ef925)|![holdnothing_4](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/d66bc05a-290f-4641-8690-2f960ced767b)|
|![holdnothing_3](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/3f4ff592-c591-4e53-90de-2670dc5a4644)|![holdnothing_2](https://github.com/SuperAI520/HandHoldingObjectClassification/assets/160987762/bc386ec4-a619-470e-bb34-41261030b8aa)|
