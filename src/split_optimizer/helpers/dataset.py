""" Kedro Torch Model IO
Models need to be imported and added to the dictionary
as shown with the ExampleModel
Example of catalog entry:
modo:
  type: kedro_example.io.torch_model.TorchLocalModel
  filepath: modo.pt
  model: ExampleModel
"""

from split_optimizer.pipelines.data_science.hybrid_model import Model
import torchvision
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple


from os.path import isfile
from typing import Any, Union, Dict
import torch
from kedro.io import AbstractDataSet


class TorchLocalModel(AbstractDataSet):
    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            model=self._model,
            load_args=self._load_args,
            save_args=self._save_args,
        )

    def __init__(
        self,
        filepath: str,
        model: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = filepath
        self._model = model
        self._Model = Model
        default_save_args = {}
        default_load_args = {}

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )
        self._save_args = (
            {**default_save_args, **save_args}
            if save_args is not None
            else default_save_args
        )

    def _load(self):
        state_dict = torch.load(self._filepath)
        model = self._Model(**self._load_args)
        model.load_state_dict(state_dict)
        return model

    def _save(self, model) -> None:
        torch.save(model.state_dict(), self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return isfile(self._filepath)


class OneHotMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, download, transform):
        super(OneHotMNIST, self).__init__(
            root=root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
