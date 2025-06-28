import logging
from pathlib import Path
import os

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False


class Logger:
    def __init__(self, use_wandb=False, project_name=None, run_name=None, config=None, name=None):
        self.use_wandb = use_wandb and _wandb_available
        self.logs = []

        if self.use_wandb:
            wandb.init(
                project=project_name or "default_project",
                name=run_name,
                config=config or {},
            )
            self.wandb = wandb
        else:
            self.logger = logging.getLogger(name or f"Logger-{os.getpid()}")
            self.logger.setLevel(logging.INFO)

            if not self.logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter("[%(name)s] %(message)s")
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

    def log(self, data: dict):
        if self.use_wandb:
            self.wandb.log(data)
        else:
            msg = ", ".join(f"{k}={v}" for k, v in data.items())
            self.logs.append(msg)
            self.logger.info(msg)

    def save_to_file(self, path, subdir="logs", idx=0, name=""): 
        log_dir = Path(path) / subdir 
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{name}_{str(idx)}.log", "w", encoding="utf-8") as f: 
            f.write("\n".join(self.logs))


    def finish(self):
        if self.use_wandb:
            self.wandb.finish()