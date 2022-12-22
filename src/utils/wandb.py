import wandb


# # Start a training run
def start_wandb_run(config: dict):
    wandb.init(project="advanced-ml", entity="7-vs-capsule", config=config)

# # Watch a model
def wandb_watch(model, *, log_freq: int = 100):
    wandb.watch(model, log_freq=log_freq)

# # Make sure to log loss and / or accuracy
def wandb_log(updates: dict):
    wandb.log(updates)