import wandb
from dataclasses import asdict, dataclass, is_dataclass

# # Start a training run
def start_wandb_run(config):
    assert is_dataclass(config)
    wandb.init(project="advanced-ml", entity="7-vs-capsule", config=asdict(config))

# # Watch a model
def wandb_watch(model, *, log_freq: int = 100):
    wandb.watch(model, log_freq=log_freq)

# # Make sure to log loss and / or accuracy
def wandb_log(updates: dict):
    wandb.log(updates)