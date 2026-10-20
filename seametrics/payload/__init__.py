"""Data carriers for evaluation payloads: resolutions, sequences and payloads."""

import logging
from dataclasses import dataclass
from typing import Dict, List

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Resolution:
    """Represents the resolution of a sequence."""

    height: int
    width: int


class Sequence:
    """Class to store sequence data.

    Allows for dynamic attributes to be added to the object.
    """

    def __init__(self, resolution: Resolution, **kwargs: object) -> None:
        """Initializes a Sequence object.

        Args:
            resolution (Resolution): The resolution of the sequence.
            **kwargs: Additional attributes to be added to the object.
        """
        self.resolution = resolution
        self.__dict__.update(kwargs)

    def __getattr__(self, attr: str) -> object:
        """Fallback lookup; raises AttributeError for a missing attribute."""
        try:
            return self.__dict__[attr]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            ) from e

    def __setattr__(self, key: str, value: object) -> None:
        self.__dict__[key] = value

    def __getitem__(self, key: str) -> object:
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: object) -> None:
        self.__setattr__(key, value)

    def __repr__(self) -> str:
        attrs = [f"{k}: {type(v).__name__}" for k, v in self.__dict__.items()]
        joined_attrs = ", ".join(attrs)
        return f"{self.__class__.__name__}({joined_attrs})"

    @property
    def field_names(self) -> List[str]:
        """Detection field names of the sequence.

        Excludes the resolution and keyframes attributes.

        Returns:
            List[str]: The list of field names.
        """
        return list(self.__dict__.keys() - {"resolution", "keyframes"})


@dataclass
class Payload:
    """Represents a payload containing dataset and model information."""

    dataset: str
    models: List[str]
    gt_field_name: str
    sequences: Dict[str, Sequence]  # key is sequence name

    @property
    def sequences_list(self) -> List[str]:
        """Sequence names in the payload.

        Returns:
            List[str]: The list of sequence names.
        """
        return list(self.sequences.keys())

    def to_dict(self) -> Dict:
        """Returns a dictionary representation of the payload.

        Returns:
            Dict: The dictionary representation of the payload.
        """
        return {
            "dataset": self.dataset,
            "models": self.models,
            "gt_field_name": self.gt_field_name,
            "sequences": {name: seq.__dict__ for name, seq in self.sequences.items()},
            "sequence_list": self.sequences_list,
        }
