from pathlib import Path
from datetime import datetime
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt 

class FileManager:
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.home() / ".uqregressors"
        self.model_dir = self.base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_timestamped_path(self, model_class_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.model_dir / f"{model_class_name}_{timestamp}"

    def get_named_path(self, name: str) -> Path:
        return self.model_dir / name

    def save_model(
        self, model, name=None, path=None, metrics=None, X_train=None, y_train=None, X_test=None, y_test=None
    ) -> Path:
        if name is not None:
            path = self.get_named_path(name)
            if path is not None:
                warnings.warn(f"Both name and path given. Using named path: {path}")
        elif path is None:
            path = self.get_timestamped_path(model.__class__.__name__)
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        if not hasattr(model, "save") or not callable(model.save):
            raise AttributeError(f"{model.__class__.__name__} must implement `save(path)`")
        model.save(path)

        if metrics:
            with open(path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        for name, array in [("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)]:
            if array is not None:
                np.save(path / f"{name}.npy", np.array(array))

        print(f"Model and additional artifacts saved to: {path}")
        return path

    def load_model(self, model_class, path=None, name=None, device="cpu", load_logs=False):
        if name:
            path = self.get_named_path(name)
        elif path:
            path = Path(path)
        else:
            raise ValueError("Either `name` or `path` must be specified.")
        
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        if not hasattr(model_class, "load") or not callable(model_class.load):
            raise AttributeError(f"{model_class.__name__} must implement `load(path)`")

        from torch.serialization import safe_globals
        with safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype]):
            model = model_class.load(path, device=device, load_logs=load_logs)

        def try_load(name):
            f = path / f"{name}.npy"
            return np.load(f) if f.exists() else None

        metrics = None
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        return {
            "model": model,
            "metrics": metrics,
            "X_train": try_load("X_train"),
            "y_train": try_load("y_train"),
            "X_test": try_load("X_test"),
            "y_test": try_load("y_test"),
        }
    
    def save_plot(self, fig, model_path: Path, filename="plot.png", show=True, subdir="plots"):
        plot_dir = model_path / subdir
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / filename
        fig.savefig(save_path, bbox_inches='tight')
        if show:
            fig.show()
        plt.close(fig)
        print(f"Plot saved to {save_path}")
        return save_path

