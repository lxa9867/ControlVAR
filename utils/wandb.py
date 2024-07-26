from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb


class CustomWandbTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False


    def __init__(self, run_name: str):
        self.run_name = run_name
        self.run = wandb.init(name=self.run_name, project='MaskVAR_local')


    def tracker(self):
        return self.run.run


    def store_init_configuration(self, values: dict):
        wandb.config.update(values)


    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)