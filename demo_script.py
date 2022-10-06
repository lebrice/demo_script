from __future__ import annotations

import datetime
import inspect
import os
import pprint
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, NewType, Protocol, TypeVar, MutableMapping
import datetime
import yaml
from mila_datamodules.vision import ImagenetDataModule, VisionDataModule
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from simple_parsing import ArgumentParser, choice, field
from simple_parsing.helpers.serialization.serializable import Serializable
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models
from typing_extensions import ParamSpec
from mila_datamodules.vision import CIFAR10DataModule
import torch
from torch import Tensor, nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

FAIRSCALE_INSTALLED = False
try:
    from fairscale.nn import auto_wrap
    from fairscale.nn.wrap.auto_wrap import ConfigAutoWrap

    FAIRSCALE_INSTALLED = True
except ImportError:
    pass

P = ParamSpec("P")


C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)
ModuleType = TypeVar("ModuleType", bound=nn.Module)


ENCODERS: dict[str, type[nn.Module]] = {
    name: cls_or_fn
    for name, cls_or_fn in vars(models).items()
    if callable(cls_or_fn)
    and any(
        f in inspect.signature(cls_or_fn).parameters for f in ["weights", "pretrained"]
    )
}


class Model(LightningModule):
    """LightningModule that uses an encoder backbone and an output layer."""

    @dataclass
    class HParams(Serializable):
        """Hyper-Parameters of the model."""

        backbone: str = choice(
            ENCODERS.keys(),  # type: ignore
            default="resnet18",
        )
        """ Choice of backbone network. """

        lr: float = field(default=3e-4)
        """ Learning rate. """

        batch_size: int = 512
        """ Batch size (in total). Gets divided evenly among the devices when using DP. """

    def __init__(
        self,
        datamodule: VisionDataModule | ImagenetDataModule,
        hp: HParams | dict | None = None,
    ) -> None:
        super().__init__()
        self._datamodule = datamodule
        assert hasattr(datamodule, "num_classes")
        num_classes = getattr(datamodule, "num_classes")
        assert isinstance(num_classes, int)
        image_dims: tuple[C, H, W] = datamodule.dims  # type: ignore
        assert len(image_dims) == 3

        self.num_classes = num_classes
        self.image_dims = image_dims
        self.hp: Model.HParams = (
            hp if isinstance(hp, Model.HParams) else self.HParams(**(hp or {}))
        )
        in_features, backbone = _get_backbone_network(
            network_type=ENCODERS[self.hp.backbone],
            image_dims=image_dims,
            pretrained=False,
        )
        self.backbone = backbone
        self.output = nn.Linear(in_features, num_classes)
        self.loss = nn.CrossEntropyLoss(reduction="none")

        self.metrics: MutableMapping[str, Metric] = nn.ModuleDict()  # type: ignore
        for phase in ["train", "val", "test"]:
            self.metrics[f"{phase}/accuracy"] = Accuracy(num_classes=num_classes)
            self.metrics[f"{phase}/top_5_accuracy"] = Accuracy(
                num_classes=num_classes, top_k=5
            )
        self.save_hyperparameters({"hp": self.hp.to_dict()})

        self.trainer: Trainer
        self._model_is_wrapped = False

    @property
    def example_input_array(self) -> Tensor:
        # NOTE: This can be useful to define, especially when trying to autoscale the batch size
        # and such. But it's a little bit buggy.
        return torch.rand([self.hp.batch_size, *self.image_dims], device=self.device)

    def configure_sharded_model(self) -> None:
        """Configures the model to be sharded for model-parallel training.

        NOTE: I think that if the model already fits inside a single GPU, this doesn't do anything.
        But I might be wrong.
        """
        super().configure_sharded_model()
        if not FAIRSCALE_INSTALLED:
            return
        # NOTE: From https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#fully-sharded-training
        # NOTE: This gets called during train / val / test, so we need to check that we don't wrap
        # the model twice.
        if not self._model_is_wrapped:
            # NOTE: Could probably use any of the cool things from fairscale here, like
            # mixture-of-experts sharding, etc!
            if ConfigAutoWrap.in_autowrap_context:
                print(f"Wrapping models for model-parallel training using fairscale")
                print(f"Trainer state: {self.trainer.state}")
            self.backbone = auto_wrap(self.backbone)
            self.output = auto_wrap(self.output)
            self._model_is_wrapped = True

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.hp.lr)

    def forward(self, x: Tensor) -> Tensor:
        h_x = self.backbone(x)
        logits = self.output(h_x)
        return logits

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        return self.shared_step(batch, batch_idx, phase="test")

    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: str
    ) -> dict:
        x, y = batch
        y = y.to(self.device)
        logits = self.forward(x)
        # partial loss (worker_batch_size, n_classes)
        loss: Tensor = self.loss(logits, y)
        return {
            "loss": loss,
            "logits": logits,
            "y": y,
        }

    def training_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        loss = self.shared_step_end(step_output, phase="train")
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step_end(self, step_output: Tensor | dict[str, Tensor]) -> Tensor:
        loss = self.shared_step_end(step_output, phase="val")
        self.log("val/loss", loss, on_epoch=True)
        return loss

    def shared_step_end(
        self, step_output: Tensor | dict[str, Tensor], phase: str
    ) -> Tensor:
        assert isinstance(step_output, dict)
        loss = step_output["loss"]  # un-reduced loss (batch_size, n_classes)
        y = step_output["y"]
        logits = step_output["logits"]
        # Log the metrics in `shared_step_end` when they are fused from all workers.

        for name, metric in self._metrics_for_phase(phase).items():
            metric(logits, y)
            self.log(name, metric, prog_bar=(phase == "train"))
        return loss.mean()

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        # TODO: If we want to add early stopping / checkpointing, we should add them here.

        return [
            EarlyStopping(monitor="val/accuracy"),
            ModelCheckpoint(
                monitor="val/accuracy",
                mode="max",
                # filename="best",  # problem is when the name does not have an identifier (epoch, metric, ...)
                # ... but that needs to be the case for automatic processing of the checkpoints.
                save_top_k=5,  # problem is when > 1
                every_n_epochs=1,
                # auto_insert_metric_name=False,
                verbose=True,
            ),
        ]

    def _metrics_for_phase(self, phase: str) -> dict[str, Metric]:
        return {
            name: metric
            for name, metric in self.metrics.items()
            if name.startswith(f"{phase}/")
        }

    # NOTE: Adding these properties in case we are using the auto_find_lr or auto_find_batch_size
    # features of the Trainer, since it modifies these attributes.

    @property
    def batch_size(self) -> int:
        return self.hp.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        print(f"Changing batch size from {self.hp.batch_size} to {value}")
        self.hp.batch_size = value
        # Also update the datamodule's batch_size attribute (they should be the same).
        self._datamodule.batch_size = value

    @property
    def lr(self) -> float:
        return self.hp.lr

    @lr.setter
    def lr(self, lr: float) -> None:
        print(f"Changing lr from {self.hp.lr} to {lr}")
        self.hp.lr = lr


def main(
    _trainer_type: type[Callable[P, Trainer]] = Trainer,
    *trainer_default_args: P.args,
    **trainer_default_kwargs: P.kwargs,
):
    """
    Runs a PyTorch-Lightning training script.

    To create the `Trainer`, the values from `trainer_default_args` and `trainer_default_kwargs`
    are used as the defaults. The values parsed from the command-line then overwrite these
    defaults.

    NOTE: trainer_type is just used so we can get nice type-checks for the trainer kwargs. We
    don't actually expect the `_trainer_type` argument to be used.

    Examples:

    Data-Parallel training:
    ```python
    main(
        gpus=torch.cuda.device_count(),
        accelerator="auto",
        strategy="dp",
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )
    ```

    Model-Parallel training:
    ```python
    main(
        gpus=torch.cuda.device_count(),
        accelerator="auto",
        strategy="fsdp",
        devices=torch.cuda.device_count(),
        max_epochs=1,
        limit_train_batches=100,
        limit_val_batches=100,
        profiler="simple",
    )
    ```
    """
    # Set some good defaults for these arguments.
    trainer_default_kwargs.setdefault("devices", "auto")
    trainer_default_kwargs.setdefault("accelerator", "auto")
    trainer_default_kwargs.setdefault("logger", False)

    model, trainer, datamodule = parse_arguments_into_components(
        _trainer_type=_trainer_type, *trainer_default_args, **trainer_default_kwargs
    )
    return run(model=model, trainer=trainer, datamodule=datamodule)


def parse_arguments_into_components(
    _trainer_type: type[Callable[P, Trainer]] = Trainer,
    *trainer_args: P.args,
    **trainer_kwargs: P.kwargs,
) -> tuple[Model, Trainer, LightningDataModule]:
    """Parses the experiment components from the command-line arguments using argparse
    (and a bit of simple-parsing).
    """

    # Create a parser
    parser = ArgumentParser(description=__doc__)

    # Add the arguments for the Model:
    parser.add_arguments(Model.HParams, "hparams")

    # Add arguments for the Trainer of PL:
    Trainer.add_argparse_args(parent_parser=parser, use_argument_group=True)
    if trainer_kwargs:
        # NOTE: Uncomment this to turn off checkpointing by default.
        # trainer_kwargs.setdefault("enable_checkpointing", False)

        # Add the given kwargs as defaults for the parser.
        parser.set_defaults(**trainer_kwargs)
        print(f"Overwriting default values for the Trainer: {trainer_kwargs}")

    args = parser.parse_args()
    args_dict = vars(args)

    hparams: Model.HParams = args_dict.pop("hparams")
    print("HParams:")
    _print_indented_yaml(asdict(hparams))

    # Rest of `args_dict` is only meant for the Trainer.
    trainer_kwargs = args_dict  # type: ignore
    callbacks = trainer_kwargs.setdefault("callbacks", [])
    assert isinstance(callbacks, list)
    print(f"Trainer kwargs:")
    _print_indented_yaml(trainer_kwargs)

    trainer_kwargs["callbacks"] = callbacks
    # pprint.pprint(trainer_kwargs)

    # Create the Trainer:
    trainer = _trainer_type(*trainer_args, **trainer_kwargs)

    datamodule = make_datamodule(batch_size=hparams.batch_size)

    model = Model(
        datamodule=datamodule,
        hp=hparams,
    )
    return model, trainer, datamodule


def run(
    model: Model,
    trainer: Trainer,
    datamodule: CIFAR10DataModule,
    evaluate_on_test_set: bool = False,
):
    """Perform a run, given the model, trainer and datamodule."""
    # NOTE: Haven't used this new method much yet. Seems to be useful when doing profiling /
    # auto-lr / auto batch-size stuff, but those don't appear to work very well anyway.
    # Leaving it here for now.
    # BUG: Seems to be causing some issues when using more than one GPU?
    # trainer.tune(model, datamodule=datamodule)

    # Train the model on the provided datamodule.
    trainer.fit(model, datamodule=datamodule)

    # NOTE: Profiler output is a big string here. We could inspect and report it if needed.
    profiler_output = trainer.profiler.summary()

    # BUG: There appears to be something weird going on when evaluating with
    # >1 GPU! In the meantime, we re-create a Trainer here just to run the evaluation.
    # TODO: actually load the best checkpoint. It's hard to do when switching trainers.
    trainer = Trainer(
        devices=1,
        accelerator="auto",
        enable_checkpointing=False,
        # resume_from_checkpoint="best",
    )
    # setup for all stages (if not done already, which is weird.)
    # TODO: Bug in CIFAR10DataModule, where .setup("validate") doesn't work.
    datamodule.prepare_data()
    datamodule.setup()
    if evaluate_on_test_set:
        # NOTE: Only do this AFTER HPO is done, otherwise your HP's are tuned on the test set!
        # TODO: When running on the test set, also log the test results.
        return trainer.test(model, datamodule, verbose=True)
    return trainer.validate(model, datamodule, verbose=True)


def _get_backbone_network(
    network_type: Callable[..., ModuleType],
    *,
    image_dims: tuple[C, H, W],
    pretrained: bool = False,
) -> tuple[int, ModuleType]:
    """Construct a backbone network using the given image dimensions and network type.

    Replaces the last fully-connected layer with a `nn.Identity`.

    Returns the dimensionality of the representations, along with the backbone network.

    TODO: Add support for more types of models from the torch hub.
    """
    backbone_signature = inspect.signature(network_type)
    if (
        "image_size" in backbone_signature.parameters
        or backbone_signature.return_annotation is models.VisionTransformer
    ):
        backbone = network_type(image_size=image_dims[-1], pretrained=pretrained)
    else:
        backbone = network_type(pretrained=pretrained)

    # Replace the output layer with a no-op, we'll create our own instead.
    if hasattr(backbone, "fc"):
        in_features: int = backbone.fc.in_features  # type: ignore
        backbone.fc = nn.Identity()
    elif isinstance(backbone, models.VisionTransformer):
        # NOTE: This is how the last few layers are created in the VisionTransformer class:
        head_layers = backbone.heads
        assert isinstance(head_layers, nn.ModuleDict)
        fc = head_layers.get_submodule("head")
        fc_index = list(head_layers).index(fc)
        assert isinstance(fc, nn.Linear)
        in_features = fc.in_features
        head_layers[fc_index] = nn.Identity()
    else:
        raise NotImplementedError(
            f"TODO: Don't yet know how to remove last fc layer(s) of networks of type "
            f"{type(backbone)}!\n"
        )

    return in_features, backbone


def make_datamodule(batch_size: int) -> VisionDataModule | ImagenetDataModule:
    """Create the LightningDataModule."""
    # TODO: Use a different datamodule if you want.
    # NOTE: This is a bit hard-coded for either the CIFAR10 or Imagenet datasets.
    # But you could switch it out. I would suggest using one of the datamodules from
    # the `mila_datamodules.vision` package, so that the dataset fetching/copying is
    # handled for you automatically.
    # datamodule = ImagenetDataModule(
    datamodule = CIFAR10DataModule(
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        # NOTE: mila_datamodules will figure out the 'right' value to use for `num_workers`.
        # num_workers=None,
    )
    return datamodule


def _print_indented_yaml(stuff):
    import textwrap
    from io import StringIO

    with StringIO() as f:
        yaml.dump(stuff, f)
        f.seek(0)
        print(textwrap.indent(f.read(), prefix="  "))


if __name__ == "__main__":
    start = datetime.datetime.now()
    results = main()
    print(results)
    t = datetime.datetime.now() - start
    print(f"Done in {t}")
