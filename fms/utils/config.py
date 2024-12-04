import copy
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import TypeVar, Union
import torch.distributed as dist

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ModelConfig")


@dataclass
class ModelConfig:
    @classmethod
    def load(cls, json_file: Union[str, os.PathLike]) -> "ModelConfig":
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return None
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)

        config = cls(
            **{
                k: v
                for k, v in json_dict.items()
                if k in inspect.signature(cls).parameters
            }
        )
        if dist.is_initialized():
            dist.barrier()
        return config

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, file_path: Union[str, os.PathLike]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        with open(file_path, "w") as f:
            json.dump(self.as_dict(), f)
        if dist.is_initialized():
            dist.barrier()

    def updated(self: T, **kwargs) -> T:
        """Clone this ModelConfig and override the parameters of the ModelConfig specified by kwargs

        Note: This will always return a deep copy

        Parameters
        ----------
        kwargs
            all possibly ModelConfig dataclass named parameters to override

        Returns
        -------
        ModelConfig
            a new instance of ModelConfig with the parameters overridden
        """
        # create a deep copy as we don't want to modify this reference
        copied_config = copy.deepcopy(self)
        unknown_params = []
        for k, v in kwargs.items():
            if hasattr(copied_config, k):
                setattr(copied_config, k, v)
            else:
                unknown_params.append(k)
        if len(unknown_params) > 0:
             if dist.is_initialized() and dist.get_rank() == 0:
                logger.info(
                    f"Found the following unknown parameters while cloning and updating the configuration: {unknown_params}"
                )
        return copied_config
