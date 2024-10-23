from typing import Any, Optional
from abc import ABC, abstractmethod
from enum import Enum
from os import makedirs
from os.path import join
from yaml import dump

class ISaveableModel:
    @abstractmethod
    def save(path: str) -> None:
        ...


class OptDirection(Enum):
    MAXIMIZE = 0
    MINIMIZE = 1

class BestModel:
    def __init__(self, metric_name: str, direction: OptDirection, save_path: str):
        self._best_value: Optional[Any] = None
        self._metric_name: str = metric_name
        self._opt_direction: OptDirection = direction
        self._save_path: str = save_path

        makedirs(self._save_path, exist_ok=True)


    def update(self, epoch: int, model, metric_value: Any) -> None:
        if self._current_model_is_better(metric_value):
            model.save_as(self._save_path, f"best_model_{self._metric_name}")

            with open(join(self._save_path, "metadata.yaml"), "w") as f:
                dump({"epoch": epoch, self._metric_name: metric_value}, f)
            
            self._best_value = metric_value


    def _current_model_is_better(self, metric_value: Any) -> bool:
        if self._best_value is None:
            return True
        
        if self._opt_direction == OptDirection.MAXIMIZE and metric_value > self._best_value:
            return True
        
        if self._opt_direction == OptDirection.MINIMIZE and metric_value < self._best_value:
            return True
            
        return False
