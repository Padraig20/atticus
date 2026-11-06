"""Small, optional W&B interface shared by the analysis stages."""

import logging
import math
from numbers import Integral, Real
import os
from pathlib import Path


def add_tracking_args(parser):
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"), help="Enable W&B in this project")
    group.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    group.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"), help="Group related pipeline runs")
    group.add_argument("--wandb-name", default=os.getenv("WANDB_NAME"))
    group.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                       default=os.getenv("WANDB_MODE", "online"),
                       help="Offline logs locally; disabled overrides the project setting")


def run_config(args):
    """Keep logging choices out of scientific configuration and cache identity."""
    return {key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items() if not key.startswith("wandb_")}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    return str(value) if isinstance(value, Path) else value


class Tracker:
    def __init__(self, args, stage):
        self.args, self.stage = args, stage
        self.mode = getattr(args, "wandb_mode", "online")
        self.project = getattr(args, "wandb_project", None)
        self.enabled = self.mode != "disabled" and bool(self.project or self.mode == "offline")
        self.run = None

    def __enter__(self):
        if self.enabled:
            try:
                import wandb
            except ImportError as error:
                raise RuntimeError("W&B logging requires the wandb dependency; install the dependencies for this stage") from error
            self.wandb = wandb
            directory = self.args.output.parent if self.stage in ("training", "prediction") else self.args.output
            directory.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(project=self.project or "atticus", mode=self.mode, job_type=self.stage,
                                  entity=getattr(self.args, "wandb_entity", None),
                                  group=getattr(self.args, "wandb_group", None),
                                  name=getattr(self.args, "wandb_name", None), dir=str(directory),
                                  config={"stage": self.stage, **run_config(self.args)},
                                  settings=wandb.Settings(console="off", disable_code=True, disable_git=True,
                                                          x_disable_stats=True))
        return self

    def configure(self, values):
        if self.run is not None:
            self.run.config.update(clean(values))

    def log(self, values):
        if self.run is not None:
            self.run.log(clean(values))

    def summarize(self, values):
        if self.run is not None:
            self.run.summary.update(clean(values))

    def table(self, name, columns, rows):
        if self.run is not None:
            self.run.log({name: self.wandb.Table(columns=list(columns), data=clean(list(rows)))})

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            try:
                self.run.finish(exit_code=1 if exc_type else 0)
            except Exception as error:
                if exc_type is None:
                    raise
                logging.warning("Could not finish W&B run: %s", error)
        return False
