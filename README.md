# Yet Another Lightning Hydra Template

Efficient workflow and reproducibility are extremely important components in every machine learning projects, which
enable to:

- Rapidly iterate over new models and compare different approaches faster.
- Promote confidence in the results and transparency.
- Save time and resources.

[PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Hydra](https://github.com/facebookresearch/hydra)
serve as the foundation upon this template. Such reasonable technology stack for deep learning prototyping offers a
comprehensive and seamless solution, allowing you to effortlessly explore different tasks across a variety of hardware
accelerators such as CPUs, multi-GPUs, and TPUs. Furthermore, it includes a curated collection of best practices and
extensive documentation for greater clarity and comprehension.

This template could be used as is for some basic tasks like Classification, Segmentation or Metric Learning, or be
easily extended for any other tasks due to high-level modularity and scalable structure.

As a baseline I have used gorgeous [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template),
reshaped and polished it, and implemented more features which can improve overall efficiency of workflow and
reproducibility.

## Quick start

```shell
# clone template
git clone https://github.com/gorodnitskiy/yet-another-lightning-hydra-template
cd yet-another-lightning-hydra-template

# install requirements
pip install -r requirements.txt
```

Or run the project in docker. See more in [Docker](#docker) section.

## Table of content

- [Yet Another Lightning Hydra Template](#yet-another-lightning-hydra-template)
  - [Quick start](#quick-start)
  - [Table of content](#table-of-content)
  - [Main technologies](#main-technologies)
  - [Project structure](#project-structure)
  - [Workflow - how it works](#workflow---how-it-works)
    - [Basic workflow](#basic-workflow)
    - [LightningDataModule](#lightningdatamodule)
    - [LightningModule](#lightningmodule)
      - [LightningModule API](#lightningmodule-api)
      - [Metrics](#metrics)
      - [Loss](#loss)
      - [Model](#model)
      - [Implemented LightningModules](#implemented-lightningmodules)
    - [Training loop](#training-loop)
    - [Evaluation and prediction loops](#evaluation-and-prediction-loops)
    - [Callbacks](#callbacks)
    - [Extensions](#extensions)
      - [DDP plugins](#ddp-plugins)
      - [GradCam](#gradcam)
  - [Hydra configs](#hydra-configs)
    - [How to run pipeline with Hydra](#how-to-run-pipeline-with-hydra)
    - [Instantiating objects with Hydra](#instantiating-objects-with-hydra)
    - [Command line operations](#command-line-operations)
    - [Additional out-of-the-box features](#additional-out-of-the-box-features)
    - [Custom config resolvers](#custom-config-resolvers)
    - [Simplify complex modules configuring](#simplify-complex-modules-configuring)
  - [Logs](#logs)
  - [Data](#data)
  - [Notebooks](#notebooks)
  - [Hyperparameters search](#hyperparameters-search)
  - [Docker](#docker)
  - [Tests](#tests)
  - [Continuous integration](#continuous-integration)

## Main technologies

[PyTorch Lightning](https://github.com/Lightning-AI/lightning) - a lightweight deep learning framework / PyTorch
wrapper for professional AI researchers and machine learning engineers who need maximal flexibility without
sacrificing performance at scale.

[Hydra](https://github.com/facebookresearch/hydra) - a framework that simplifies configuring complex applications.
The key feature is the ability to dynamically create a hierarchical configuration by composition and override it
through config files and the command line.

## Project structure

The structure of a machine learning project can vary depending on the specific requirements and goals of the project,
as well as the tools and frameworks being used. However, here is a general outline of a common directory structure for
a machine learning project:

- `src/`
- `data/`
- `logs/`
- `tests/`
- some additional directories, like: `notebooks/`, `docs/`, etc.

In this particular case, the directory structure looks like:

```
├── configs                     <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── debug                   <- Debugging configs
│   ├── experiment              <- Experiment configs
│   ├── extras                  <- Extra utilities configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra settings configs
│   ├── local                   <- Local configs
│   ├── logger                  <- Logger configs
│   ├── module                  <- Module configs
│   ├── paths                   <- Project paths configs
│   ├── trainer                 <- Trainer configs
│   │
│   ├── eval.yaml               <- Main config for evaluation
│   └── train.yaml              <- Main config for training
│
├── data                        <- Project data
├── logs                        <- Logs generated by hydra, lightning loggers, etc.
├── notebooks                   <- Jupyter notebooks.
├── scripts                     <- Shell scripts
│
├── src                         <- Source code
│   ├── callbacks               <- Additional callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── modules                 <- Lightning modules
│   ├── utils                   <- Utility scripts
│   │
│   ├── eval.py                 <- Run evaluation
│   └── train.py                <- Run training
│
├── tests                       <- Tests of any kind
│
├── .dockerignore               <- List of files ignored by docker
├── .gitattributes              <- List of additional attributes to pathnames
├── .gitignore                  <- List of files ignored by git
├── .pre-commit-config.yaml     <- Configuration of pre-commit hooks for code formatting
├── Dockerfile                  <- Dockerfile
├── Makefile                    <- Makefile with commands like `make train` or `make test`
├── pyproject.toml              <- Configuration options for testing and linting
├── requirements.txt            <- File for installing python dependencies
├── setup.py                    <- File for installing project as a package
└── README.md
```

## Workflow - how it works

Before starting a project, you need to think about the following things to unsure in results reproducibility:

- Docker image setting up
- Freezing python package versions
- Code Version Control
- Data Version Control. Many of which currently provide not just Data Version Control, but a lot of side very useful
  features like Model Registry or Experiments Tracking:
  - [DVC](https://dvc.org)
  - [Neptune](https://neptune.ai)
  - Your own solution or others...
- Experiments Tracking tools:
  - [Weights & Biases](https://wandb.ai)
  - [Neptune](https://neptune.ai)
  - [DVC](https://dvc.org)
  - [Comet](https://www.comet.com/)
  - [MLFlow](https://mlflow.org)
  - TensorBoard
  - Or just CSV files...

### Basic workflow

This template could be used as is for some basic tasks like Classification, Segmentation or Metric Learning approach,
but if you need to do something more complex, here it is a general workflow:

1. Write your PyTorch Lightning DataModule (see examples in [datamodules/datamodules.py](src/datamodules/datamodules.py))
2. Write your PyTorch Lightning Module (see examples in [modules/single_module.py](src/modules/single_module.py))
3. Fill up your configs, in particularly create experiment configs
4. Run experiments:
   - Run training with chosen experiment config:
   ```shell
   python src/train.py experiment=experiment_name.yaml
   ```
   - Use hyperparameter search, for example by Optuna Sweeper via Hydra:
   ```shell
   # using Hydra multirun mode
   python src/train.py -m hparams_search=mnist_optuna
   ```
   - Execute the runs with some config parameter manually:
   ```shell
   python src/train.py -m logger=csv module.optimizer.weight_decay=0.,0.00001,0.0001
   ```
5. Run evaluation with different checkpoints or prediction on custom dataset for additional analysis

The template contains example with `MNIST` classification, which uses for tests by the way.
If you run `python src/train.py`, you will get something like this:

<details>

<summary><b>Show terminal screen when running pipeline</b></summary>

![Terminal screen when running pipeline](https://user-images.githubusercontent.com/8071747/217344340-a336c01c-8b8c-48c4-beef-c5e1478537e4.jpg)

</details>

### LightningDataModule

At the start, you need to create PyTorch Dataset for you task. It has to include `__getitem__` and `__len__` methods.
Maybe you can use as is or easily modify already [implemented Datasets](src/datamodules/datasets.py) in the template.
See more details in [PyTorch documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Also, it could be useful to see [section](#data) about how it is possible to save data for training and evaluation.

Then, you need to create DataModule using [PyTorch Lightning DataModule API](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#lightningdatamodule-api).
By default, API has the following methods:

- `prepare_data` (optional): perform data operations on CPU via a single process, like load and preprocess data, etc.
- `setup` (optional): perform data operations on every GPU, like train/val/test splits, create datasets, etc.
- `train_dataloader`: used to generate the training dataloader(s)
- `val_dataloader`: used to generate the validation dataloader(s)
- `test_dataloader`: used to generate the test dataloader(s)
- `predict_dataloader` (optional): used to generate the prediction dataloader(s)

<details>

<summary><b>Show LightningDataModule API</b></summary>

```python
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


class YourDataModule(LightningDataModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.train_set: Optional[Dataset] = None
        self.valid_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None
        self.predict_set: Optional[Dataset] = None
        ...

    def prepare_data(self) -> None:
        # (Optional) Perform data operations on CPU via a single process
        # - load data
        # - preprocess data
        # - etc.
        ...

    def setup(self, stage: str) -> None:
        # (Optional) Perform data operations on every GPU:
        # - count number of classes
        # - build vocabulary
        # - perform train/val/test splits
        # - create datasets
        # - apply transforms (which defined explicitly in your datamodule)
        # - etc.
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = ...
            self.valid_set = ...
            self.test_set = ...
        if (stage == "predict") and not self.predict_set:
            self.predict_set = ...

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        # Used to generate the training dataloader(s)
        # This is the dataloader that the Trainer `fit()` method uses
        return DataLoader(self.train_set, ...)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # Used to generate the validation dataloader(s)
        # This is the dataloader that the Trainer `fit()` and `validate()` methods uses
        return DataLoader(self.valid_set, ...)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # Used to generate the test dataloader(s)
        # This is the dataloader that the Trainer `test()` method uses
        return DataLoader(self.test_set, ...)

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # Used to generate the prediction dataloader(s)
        # This is the dataloader that the Trainer `predict()` method uses
        return DataLoader(self.predict_set, ...)

    def teardown(self, stage: str) -> None:
        # Used to clean-up when the run is finished
        ...
```

</details>

See examples of `datamodule` configs in [configs/datamodule](configs/datamodule) folder.

By default, the template contains the following DataModules:

- [SingleDataModule](src/datamodules/datamodules.py) in which `train_dataloader`, `val_dataloader` and
  `test_dataloader` return single DataLoader, `predict_dataloader` returns list of DataLoaders
- [MultipleDataModule](src/datamodules/datamodules.py) in which `train_dataloader` return dict of DataLoaders,
  `val_dataloader`,  `test_dataloader` and `predict_dataloader` return list of DataLoaders

In the template, DataModules has `_get_dataset_` method to simplify Datasets instantiation.

### LightningModule

#### LightningModule API

Next, your need to create LightningModule using [PyTorch Lightning LightningModule API](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).
Minimum API has the following methods:

- `forward`: use for inference only (separate from training_step)
- `training_step`: the complete training loop
- `validation_step`: the complete validation loop
- `test_step`: the complete test loop
- `predict_step`: the complete prediction loop
- `configure_optimizers`: define optimizers and LR schedulers

Also, you can override optional methods for each step to perform additional logic:

- `training_step_end`: training step end operations
- `on_train_epoch_end`: training epoch end operations
- `validation_step_end`: validation step end operations
- `on_validation_epoch_end`: validation epoch end operations
- `test_step_end`: test step end operations
- `on_test_epoch_end`: test epoch end operations

<details>

<summary><b>Show LightningModule API methods and appropriate order</b></summary>

```python
from typing import Any
from pytorch_lightning import LightningModule


class LitModel(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        ...

    def forward(self, *args: Any, **kwargs: Any):
        ...

    def training_step(self, *args: Any, **kwargs: Any):
        ...

    def training_step_end(self, step_output: Any):
        ...

    def on_train_epoch_end(self, outputs: Any):
        ...

    def validation_step(self, *args: Any, **kwargs: Any):
        ...

    def validation_step_end(self, step_output: Any):
        ...

    def on_validation_epoch_end(self, outputs: Any):
        ...

    def test_step(self, *args: Any, **kwargs: Any):
        ...

    def test_step_end(self, step_output: Any):
        ...

    def on_test_epoch_end(self, outputs: Any):
        ...

    def configure_optimizers(self):
        ...

    def any_extra_hook(self, *args: Any, **kwargs: Any):
        ...
```

</details>

In the template, LightningModule has `model_step` method to adjust repeated operations, like `forward` or `loss`
calculation, which are required in `training_step`, `validation_step` and `test_step`.

#### Metrics

The template offers the following `Metrics API`:

- `main` metric: main metric, which also uses for all callbacks or trackers like `model_checkpoint`, `early_stopping`
  or `scheduler.monitor`.
- `valid_best` metric: use for tracking the best validation metric. Usually it can be `MaxMetric` or `MinMetric`.
- `additional` metrics: additional metrics.

Each metric config should contain `_target_` key with metric class name and other parameters which are required by
metric. The template allows to use any metrics, for example from
[torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) or implemented by yourself (see examples in
`modules/metrics/components/` or [torchmetrics API](https://torchmetrics.readthedocs.io/en/latest/references/metric.html)).

See more details about implemented [Metrics API](src/modules/metrics/metrics.py) and `metrics` config as a part of
`network` configs in [configs/module/network](configs/module/network) folder.

Metric config example:

```yaml
metrics:
  main:
    _target_: "torchmetrics.Accuracy"
    task: "binary"
  valid_best:
    _target_: "torchmetrics.MaxMetric"
  additional:
    AUROC:
      _target_: "torchmetrics.AUROC"
      task: "binary"
```

Also, the template includes few manually implemented metrics:

- [`Accuracy`](src/modules/metrics/components/classification.py)
- [`NDCG`](src/modules/metrics/components/classification.py)
- [`MRR`](src/modules/metrics/components/classification.py)
- [`SentiMRR`](src/modules/metrics/components/classification.py)
- [`PrecisionAtRecall`](src/modules/metrics/components/classification.py)
- [`IoU`](src/modules/metrics/components/segmentation.py)

#### Loss

The template offers the following `Losses API`:

- Loss config should contain `_target_` key with loss class name and other parameters which are required by loss.
- Parameter contains `weight` string in name will be wrapped by `torch.tensor` and cast to `torch.float` type before
  passing to loss due to requirements from most of the losses.

The template allows to use any losses, for example from
[PyTorch](https://pytorch.org/docs/stable/nn.html#loss-functions) or implemented by yourself (see examples in
`modules/losses/components/`).

See more details about implemented [Losses API](src/modules/losses/losses.py) and `loss` config as a part of
`network` configs in [configs/module/network](configs/module/network) folder.

Loss config examples:

```yaml
loss:
  _target_: "torch.nn.CrossEntropyLoss"
```

```yaml
loss:
  _target_: "torch.nn.BCEWithLogitsLoss"
  pos_weight: [0.25]
```

```yaml
loss:
  _target_: "src.modules.losses.VicRegLoss"
  sim_loss_weight: 25.0
  var_loss_weight: 25.0
  cov_loss_weight: 1.0
```

Also, the template includes few manually implemented losses:

- [`VicRegLoss`](src/modules/losses/components/vicreg_loss.py) as example for self-supervised learning
- [`FocalLoss`](src/modules/losses/components/focal_loss.py): use for `Extremely Imbalanced` tasks
- [`AngularPenaltySMLoss`](src/modules/losses/components/margin_loss.py): use for `Metric Learning` approach

#### Model

The template offers the following `Model API`, model config should contain:

- `_target_`: key with model class name.
- `model_name`: model name
- `model_repo` (optional): model repository
- Other parameters which are required by model.

By default, model can be loaded from:

- [torchvision.models](https://pytorch.org/vision/stable/models.html) with setting up `model_name` as
  `torchvision.models/<model-name>`, for example `torchvision.models/mobilenet_v3_large`
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch/) with setting up `model_name` as
  `segmentation_models_pytorch/<model-name>`, for example `segmentation_models_pytorch/Unet`
- [timm](https://rwightman.github.io/pytorch-image-models/) with setting up `model_name` as `timm/<model-name>`, for
  example `timm/mobilenetv3_100`
- [torch.hub](https://pytorch.org/hub/) with setting up `model_name` as `torch.hub/<model-name>` and `model_repo`, for
  example `model_name="torch.hub/resnet18"` and `model_repo="pytorch/vision"`

See more details about implemented [Model API](src/modules/models) and `model` config as a part of `network` configs in
[configs/module/network](configs/module/network) folder.

Model config example:

```yaml
model:
  _target_: "src.modules.models.classification.Classifier"
  model_name: "torchvision.models/mobilenet_v3_large"
  model_repo: null
  weights: "IMAGENET1K_V2"
  num_classes: 1
```

#### Implemented LightningModules

By default, the template contains the following LightningModules:

- [SingleLitModule](src/modules/single_module.py) contains LightningModules for a few tasks, like ordinary,
  self-supervised learning and metric learning approach, which require single DataLoader on each step
- [MultipleLitModule](src/datamodules/datamodules.py) contains LightningModules, which require multiple DataLoaders on
  each step

See examples of `module` configs in [configs/module](configs/module) folder.

<details>

<summary><b>Show LightningModule config example</b></summary>

```yaml
_target_: src.modules.single_module.MNISTLitModule

defaults:
  - _self_
  - network: mnist.yaml

optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0

scheduler:
  scheduler:
    _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
    mode: "max"
    factor: 0.1
    min_lr: 1.0e-9
    patience: 10
    verbose: True
  extras:
    monitor: ${replace:"__metric__/valid"}
    interval: "epoch"
    frequency: 1

logging:
  on_step: False
  on_epoch: True
  sync_dist: False
  prog_bar: True
```

</details>

### Training loop

[Training loop](src/train.py) in the template contains the following stages:

- LightningDataModule instantiating
- LightningModule instantiating
- Callbacks instantiating
- Loggers instantiating
- Plugins instantiating
- Trainer instantiating
- Hyperparameteres and metadata logging
- Training the model
- Testing the best model

See more details in [training loop](src/train.py) and [configs/train.yaml](configs/train.yaml).

### Evaluation and prediction loops

[Evaluation loop](src/eval.py) in the template contains the following stages:

- LightningDataModule instantiating
- LightningModule instantiating
- Loggers instantiating
- Trainer instantiating
- Hyperparameteres and metadata logging
- Evaluating model or predicting

See more details in [evaluation loop](src/eval.py) and [configs/eval.yaml](configs/eval.yaml).

The template contains the following [Prediction API](src/utils/saving_utils.py):

- Set `predict: True` in `configs/eval.yaml` to turn on prediction mode.
- DataModule could contain multiple predict datasets:
  ```yaml
  datasets:
    predict:
      dataset1:
        _target_: src.datamodules.datasets.ClassificationDataset
        json_path: ${paths.data_dir}/predict/data1.json
      dataset2:
        _target_: src.datamodules.datasets.ClassificationDataset
        json_path: ${paths.data_dir}/predict/data2.json
  ```
- PyTorch Lightning returns a list of batch predictions, when `LightningDataModule.predict_dataloader()` returns a
  single dataloader, and a list of lists of batch predictions, when `LightningDataModule.predict_dataloader()` returns
  multiple dataloaders.
- Predictions log to `{cfg.paths.output_dir}/predictions/` folder.
- If there are a multiple predict dataloaders, predictions will be saved with `_<dataloader_idx>` postfix. It isn't
  possible to use dataset names due to PyTorch Lightning doesn't allow to return a dict of dataloaders from
  `LightningDataModule.predict_dataloader()` method.
- There are two possible built-in output formats: `csv` and `json`. `json` format is used by default, but it might be
  more effectively to use `csv` format for a large number of predictions, it could help to avoid RAM memory overflow,
  because `csv` allows to write row by row and doesn't require to keep in RAM the whole dict like in case of `json`.
  To change output format, set `predictions_saving_params.output_format` variable in `configs/extra/default.yaml`
  config file.
- If you need some custom output format, for example `parquet`, you can easily modify
  `src.utils.saving_utils.save_predictions()` method.

See more details about [Prediction API](src/utils/saving_utils.py) and
[`predict_step` in LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_basic.html).

### Callbacks

PyTorch Lightning has a lot of [built-in callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html),
which can be used just by adding them to callbacks config, thanks to Hydra.
See examples in [callbacks config](configs/callbacks) folder.

By default, the template contains few of them:

- Model Checkpoint
- Early Stopping
- Model Summary
- Rich Progress Bar

However, there is an additional `LightProgressBar` callback, which might be more beautiful and useful, instead of using
`RichProgressbar`:

![LightProgressBar preview](https://user-images.githubusercontent.com/8071747/217344860-987e7ee4-f507-4df0-a0b1-2c1d81e7ea29.png)

### Extensions

#### DDP plugins

Lightning provides [DDP plugins](https://pytorch-lightning.readthedocs.io/en/stable/extensions/plugins.html) which allow
custom integrations to the internals of the Trainer such as custom precision, checkpointing or cluster environment
implementation.

#### GradCam

By default, the template provides [GradCam](https://github.com/jacobgil/pytorch-grad-cam) package which can be extremely
useful for understanding network decision and explainability in general.

See more details in [package documentation](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html).

## Hydra configs

[Hydra](https://github.com/facebookresearch/hydra) + [OmegaConf](https://omegaconf.readthedocs.io/en/) provide a
flexible and efficient configuration management system that allows to dynamically create a hierarchical configurations
by composition and override it through config files and the command line.

It is core component of the template which is orchestrating all other modules.

This powerful tools allow to create a simple and efficient way for managing and organizing the various configurations
in one place, constructing complex configurations structure without any limits which can be essential in machine
learning projects.

All of that enable to easily switch between any parameters and try different configurations without having to manually
update the code.

### How to run pipeline with Hydra

A decorator `hydra.main` is supposed to be used to load Hydra config during launching of the pipeline. Here a config
is being parser by Hydra grammar parser, merged, composed and passed to the pipeline main function.

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import train


@hydra.main(version_base="1.3", config_path=".", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
```

Also, it could be done by Hydra [`Compose API`](https://hydra.cc/docs/advanced/compose_api/), use
`initialize`, `initialize_config_module` or `initialize_config_dir`, instead of `hydra.main`:

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from src.train import train


def main() -> None:
    with initialize_config_dir(version_base="1.3", config_dir="."):
        cfg = compose(
            config_name="train.yaml", return_hydra_config=True, overrides=[]
        )
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
```

### Instantiating objects with Hydra

For object instantiating from a config, it should contain `_target_` key with `class` or `function` name which
would be instantiated with other parameters passed to config. Hydra provides `hydra.utils.instantiate()` (and its alias
`hydra.utils.call()`) for instantiating objects and calling `class` or `function`. Prefer `instantiate` for creating
objects and `call` for invoking functions.

```yaml
loss:
  _target_: "torch.nn.CrossEntropyLoss"

metric:
  _target_: "torchmetrics.Accuracy"
  task: "multiclass"
  num_classes: 10
  top_k: 1
```

Based on such config you could instantiate loss via `loss = hydra.utils.instantiate(config.loss)` and metric via
`metric = hydra.utils.instantiate(config.metric)`.

It supports few config [parameters conversion strategies](https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies):

- `none`: default behavior, use OmegaConf containers
- `partial`: convert OmegaConf containers to dict and list, except Structured Configs, which remain as DictConfig
  instances.
- `object`: convert OmegaConf containers to dict and list, except Structured Configs, which are converted to instances
  of the backing dataclass / attr class using OmegaConf.to_object
- `all`: convert everything to primitive containers

It is managed by adding an additional `_convert_` parameter to config.

Moreover, it offers a [partial instantiation](https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation),
which can be very useful, for example for function instantiation or recursively object instantiation. It is managed by
adding an additional `_partial_` bool parameter to config.

```yaml
output_activation:
  _target_: "torch.softmax"
  dim: 1
```

```
output_activation = hydra.utils.instantiate(
    config.output_activation, _partial_=True
)
preds = output_activation(logits)
```

### Command line operations

It supports few operations from the command line as well:

- Override existing config value by passing it as well
- Add a new config value, which doesn't exist in the config, by using `+`
- Override a config value if it's already in the config, or add it otherwise, by using `++`

```shell
# train the model with the default config
python src/train.py

# train the model with the overridden parameter
python src/train.py model.model_name="torchvision.models/vit_l_16"

# train the model with the overridden parameter and add a new parameter
python src/train.py model.model_name="torch.hub/vit_l_16_v2" ++model.model_repo="repository"
```

### Additional out-of-the-box features

It provides much more excited features like:

- [Structured configs](https://hydra.cc/docs/tutorials/structured_config/schema/) with extended list of available
  primitive types, nested structure, containers containing primitives, default values, bottom-up values overriding and
  much more. It gives even more possibilities to structure configs in any ways.

```yaml
_target_: src.datamodules.datamodules.SingleDataModule

defaults:
  - _self_
  - loaders: default.yaml
  - transforms: default.yaml

datasets:
  train:
    _target_: src.datamodules.datasets.ClassificationDataset
    json_path: ${paths.data_dir}/train/data.json

  valid:
    _target_: src.datamodules.datasets.ClassificationDataset
    json_path: ${paths.data_dir}/valid/data.json

  test:
    _target_: src.datamodules.datasets.ClassificationDataset
    json_path: ${paths.data_dir}/test/data.json
```

- Colored logs for `hydra/job_logging` and `hydra/hydra_logging`
  ![Colorlog](https://hydra.cc/assets/images/colorlog-b20147697b9d16362f62a5d0bb58347f.png)
- Hyperparameters sweepers: [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/),
  [Nevergrad](https://hydra.cc/docs/plugins/nevergrad_sweeper/), [Ax](https://hydra.cc/docs/plugins/ax_sweeper/)
- [Custom plugins](https://hydra.cc/docs/advanced/plugins/develop/)

### Custom config resolvers

Hydra provides a way to extend furthermore its functionality by adding custom resolvers via
`OmegaConf.register_new_resolver()`. It allows to add custom executable expressions to the config.
See more [here](https://omegaconf.readthedocs.io/en/2.1_branch/custom_resolvers.html#built-in-resolvers).

By default, OmegaConf supports the following resolvers:

- `oc.env`: returns the value of an environment variable
- `oc.create`: may be used for dynamic generation of config nodes
- `oc.deprecated`: may be used to mark a config node as deprecated
- `oc.decode`: decodes a string using a given codec
- `oc.select`: provides a default value to use in case the primary interpolation key is not found, or select keys that
  are otherwise illegal interpolation keys, or works with missing values
- `oc.dict.{keys,value}`: analogous to the `dict.keys` and `dict.values` methods in plain Python dictionaries

But it is a powerful tool, which allows to add any other custom resolvers! For example, it could be annoying to write
loss or metric names in configs in each place where it might be required, like `early_stopping` config,
`model_checkpoint` config, config contained scheduler params, or somewhere else. It could be solved by adding custom
resolver for replacing `__loss__` and `__metric__` names by the actual loss or metric name, which is passed to the
config and initialized by Hydra.

> **Note**: You need to register custom resolvers before `hydra.main` or `Compose API` calls. Otherwise, it just
> doesn't apply by Hydra config parser.

In my template, it is implemented as a decorator `utils.register_custom_resolvers`, which allows to register all custom
resolvers in a single place. It supports Hydra's command line flags, which are required to override config path, name or
dir. By default, it allows to replace `__loss__` to `loss.__class__.__name__` and `__metric__` to
`main_metric.__class__.__name__` via such syntax: `${replace:"__metric__/valid"}`. Use quotes for defining internal
value in `${replace:"..."}` to avoid grammar problems with hydra config parser. It can be easily expanded for any other
purposes.

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from src.train import train
from src import utils

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": ".",
    "config_name": "train.yaml",
}

@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
```

### Simplify complex modules configuring

This powerful tool allows to significantly simplify pipeline developing and configuring, for example:

- Instantiate modules with any custom logic under the hood, eg:
  - Instantiate the whole module with all inside submodules recursively by Hydra
  - Main module and some part of inside submodules can be initialized by Hydra and rest of them manually
  - Manually initialize main module and all submodules
- Package dynamic structures, like data augmentations to config, where you can easily set up any transforms classes,
  parameters or applying order. See more about implemented [TransformsWrapper](src/datamodules/components/transforms.py),
  which can be easily reworked for any over augmentations package. Config example:

```yaml
train:
  order:
    [
      "resize",
      "random_brightness_contrast",
      "normalize",
    ]
  resize:
    _target_: albumentations.Resize
    height: 256
    width: 256
    p: 1.0
  random_brightness_contrast:
    _target_: albumentations.RandomBrightnessContrast
    brightness_limit: [-0.2, 0.2]
    contrast_limit: [-0.2, 0.2]
    p: 0.5
  normalize:
    _target_: albumentations.Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    p: 1.0
```

- And many other things... It has very low limitations, so you can use it for any custom logic, which you want to use
  in your project.

## Logs

Hydra creates new output directory in `logs/` for every executed run.

Furthermore, template offers to save additional metadata for better reproducibility and debugging, including:

- `pip` logs
- `git` logs
- environment logs: CPU, GPU (nvidia-smi)
- full copy of `src/` and `configs/` directories

Default logging structure:

```
├── logs
│   ├── task_name
│   │   ├── runs                        <- Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS     <- Datetime of the run
│   │   │   │   ├── .hydra              <- Hydra logs
│   │   │   │   ├── csv                 <- Csv logs
│   │   │   │   ├── wandb               <- Weights&Biases logs
│   │   │   │   ├── checkpoints         <- Training checkpoints
│   │   │   │   ├── metadata            <- Metadata
│   │   │   │   │   ├── pip.log         <- Pip logs
│   │   │   │   │   ├── git.log         <- Git logs
│   │   │   │   │   ├── env.log         <- Environment logs
│   │   │   │   │   ├── src             <- Full copy of `src/` directory
│   │   │   │   │   └── configs         <- Full copy of `configs/` directory
│   │   │   │   └── ...                 <- Any other thing saved during training
│   │   │   └── ...
│   │   │
│   │   └── multiruns                   <- Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS     <- Datetime of the multirun
│   │       │   ├──1                    <- Multirun job number
│   │       │   ├──2
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                          <- Logs generated when debugging config is attached
│       └── ...
```

You can change this structure by modifying paths in [Hydra configuration](configs/hydra/default.yaml) and set another
path constants in [path configuration](configs/paths/default.yaml).

## Data

Usually, images or any other data files just storage on disk in folders. It is simple and convenient way.

However, there are another methods and one of them calls as [Hierarchical Data Format HDF5](https://docs.h5py.org/en/stable/)
or h5py, which has few reasons why it might be more beneficial to store images in HDF5 files instead of just folders:

- Efficient storage: the data format is designed specifically for storing large amounts of data. It is particularly
  well-suited for storing arrays of data, like images, and can compress the data to reduce the overall size of the file.
  The important thing about compressing in HDF5 files is that objects are compressed independently and only the objects
  that you need get decompressed on output. This is clearly more efficient than compressing the entire file and having to
  decompress the entire file to read it.
- Fast access: HDF5 allows you to access the data stored in the file using indexing, just like you would with a NumPy
  array. This makes it easy and fast to retrieve the data you need, which can be especially important when you are
  working with large datasets.
- Easy to use: HDF5 is easy to use and integrates well with other tools commonly used in machine learning, such as
  NumPy and PyTorch. This means you can use HDF5 to store your data and then load it into your training code without any
  additional preprocessing.
- Self-describing: it is possible to add information that helps users and tools know what is in the file. What are the
  variables, what are their types, what tools collected and wrote them, etc. The tool you are working on can read
  metadata for files. Attributes in an HDF5 file can be attached to any object in the file – they are not just file
  level information.

This template contains a tool which might be used to easily create and read HDF5 files.

To create HDF5 file:

```python
from src.datamodules.components.h5_file import H5PyFile

H5PyFile().create(
    filename="/path/to/dataset_train_set_v1.h5",
    content=["/path/to/image_0.png", "/path/to/image_1.png", ...],
    # each content item loads as np.fromfile(filepath, dtype=np.uint8)
)
```

To read HDF5 file in the wild:

```python
import matplotlib.pyplot as plt
from src.datamodules.components.h5_file import H5PyFile

h5py_file = H5PyFile(filename="/path/to/dataset_train_set_v1.h5")
image = h5py_file[0]

plt.imshow(image)
```

To read HDF5 file in `Dataset.__getitem__`:

```
def __getitem__(self, index: int) -> Any:
    key = self.keys[index]  # get the image key, e.g. path
    data_file = self.data_file
    source = data_file[key]  # get the image
    image = io.BytesIO(source)  # read the image
    ...
```

## Notebooks

Jupyter Notebook is a powerful tool for data analysis, visualization and presenting Machine Learning projects.
Such sections formatting can be used to have clean and understandable structure of a jupyter notebook:

```
├── Summary                 <- Summary: for fastly understanding what it is about
│   ├── Objective           <- Objective of analysis
│   ├── Methods             <- Methods which are using
│   └── Results             <- Results of analysis
│
├── Config                  <- Configs and constants
│
├── Libs                    <- Imports and environment variables defining
│
└── Analysis                <- Analysis / Visualization / Presenting / etc.
```

Furthermore, it can be helpful to use special naming convention for jupyter notebooks: a number (for ordering), the
creator's initials and a short description, all of this delimited by `-`, e.g. `1.0-asg-data-exploration.ipynb`.

## Hyperparameters search

Hydra provides out-of-the-box hyperparameters sweepers: [Optuna, Nevergrad or Ax](https://hydra.cc/docs/plugins/optuna_sweeper/).

You can define hyperparameters search by adding new config file to [configs/hparams_search](configs/hparams_search).
See example of [hyperparameters search config](configs/hparams_search/mnist_optuna.yaml). With this method, there is no
need to add extra code, everything is specified in a single configuration file. The only requirement is to return the
optimized metric value from the launch file.

Execute it with:

```shell
python src/train.py hparams_search=mnist_optuna
```

The `optimization_results.yaml` will be available under `logs/task_name/multirun` folder.

## Docker

Docker is essential part of environment reproducibility that makes it possible to easily package a machine learning
pipeline and its dependencies into a single container that can be easily deployed and run on any environment.
This is particularly useful due to it helps to ensure that the code will run consistently, regardless of the
environment in which it is deployed.

Docker image could require some additional packages depends on which device is used for running. For example,
for running on cluster with NVIDIA GPUs it requires the CUDA Toolkit from NVIDIA. The CUDA Toolkit provides everything
you need to develop GPU-accelerated applications, including GPU-accelerated libraries, a compiler, development tools
and the CUDA runtime.

In general, there are many way how to set up it, but to simplify this process you can use:

- [Official Nvidia Docker Images Hub](https://hub.docker.com/r/nvidia/cuda/tags), where it is easy to find images
  with any combinations of OS, CUDA, etc. See more on [`Dockerfile`](Dockerfile) and [`.dockerignore`](.dockerignore).
- Miniconda for GPU environments.

Moreover, it can be advantageous to use:

- Additional docker container runtime options for [managing resources constraints](https://docs.docker.com/config/containers/resource_constraints/), like `--cpuset-cpus`, `--gpus`, etc.
- [NVTOP](https://github.com/Syllo/nvtop) - a (h)top like task monitor for AMD, Intel and NVIDIA GPUs.

![NVTOP interface](https://user-images.githubusercontent.com/8071747/217345317-99f92914-c568-4c1a-9c54-053ace0315bc.png)

Here it is some example of container running based on [`Dockerfile`](Dockerfile) and [`.dockerignore`](.dockerignore):

```shell
set -o errexit
export DOCKER_BUILDKIT=1
export PROGRESS_NO_TRUNC=1

docker build --tag <project-name> \
    --build-arg OS_VERSION="22.04" \
    --build-arg CUDA_VERSION="11.7.0" \
    --build-arg PYTHON_VERSION="3.10" \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --build-arg NAME="<your-name>" \
    --build-arg WORKDIR_PATH=$(pwd) .

docker run \
    --name <task-name> \
    --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd):$(pwd):rw \
    --gpus '"device=0,1,3,4"' \
    --cpuset-cpus "0-47" \
    -it \
    --entrypoint /bin/bash \
    <project-name>:latest
```

## Tests

Tests are an important aspect of software development in general, and especially in Machine Learning, because here it
can be much more difficult to understand if code are working correctly without testing. Consequently, template contains
some generic tests implemented with [`pytest`](https://docs.pytest.org/en/7.2.x/).

For this purpose MNIST is used. It is a small dataset, so it is possible to run all tests on CPU. However, it is easy
to implement tests for your own dataset if it requires.

As a baseline the tests cover:

- Main module configs instantiation by Hydra
- DataModule
- Losses loading
- Metrics loading
- Models loading and utils
- Training on 1% of MNIST dataset, for example:
  - running 1 train, val and test steps
  - running 1 epoch, saving checkpoint and resuming for the second epoch
  - running 2 epochs with DDP simulated on CPU
- Evaluating and predicting
- Hyperparameters optimization
- Custom progress bar functionality
- Utils

All this implemented tests created for verifying that the main pipeline modules and utils are executable and working as
expected in general. However, sometimes it couldn't be enough to ensure that the code is working correctly, especially
in case of more complex pipelines and models.

For running:

```shell
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run tests from specific test
pytest tests/test_train.py::test_train_ddp_sim

# run all tests except the ones marked as slow
pytest -k "not slow"
```

## Continuous integration

Template contains few initial CI workflows via GitHub Actions platform. It makes it easy to automate and streamline
development workflows, which can help to save time and efforts, increase efficiency, and improve overall quality of the
code. In particularly, it includes:

- `.github/workflows/test.yaml`: running all tests from `tests/` with `pytest` on `Linux`, `Mac` and `Windows` platforms
- `.github/workflows/code-quality-main.yaml`: running `pre-commits` on main branch for all files
- `.github/workflows/code-quality-pr.yaml`: running `pre-commits` on pull requests for modified files only

> **Note**: You need to enable the GitHub Actions from the settings in your repository.

See more about [GitHub Actions for CI](https://docs.github.com/en/actions/learn-github-actions/introduction-to-github-actions).

In case of using GitLab, it is easy to set up [GitLab CI](https://docs.gitlab.com/ee/ci/) based on GitHub Actions
workflows. Here it manages by `.gitlab-ci.yml` file. See more [here](https://docs.gitlab.com/ee/ci/yaml/gitlab_ci_yaml.html).
