# MS-2D-Segmentation


## Installation
To set up the project, please follow these steps:

1. Clone this repository:
```bash
git clone https://github.com/TeaRelaxation/MS-2D-Segmentation.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from
[source link](https://link.com)
and place it in the following directory:
```bash
datasets/MS/
```

4. To train the model, run the following command with:
```bash
python ./scripts/train.py
  --experiment_name=MyExperiment\
  --model_name=UNet\
  --dataset_name=MS\
  --dataset_path=../datasets/MS\
  --logs_path=../logs\
  --n_classes=5\
  --batch_size=4\
  --epochs=10\
  --learning_rate=0.001
```

Refer to the source code for additional options and configurations.
