import numpy as np 
import torch
from sklearn.model_selection import train_test_split 
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.utils.validate_dataset import clean_dataset, validate_dataset
from uqregressors.metrics.metrics import compute_all_metrics
from uqregressors.utils.data_loader import load_unformatted_dataset
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from pathlib import Path
import pickle


def test_regressor(model, X, y, dataset_name, test_size, seed=42): 
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = clean_dataset(X, y)
    validate_dataset(X, y, name=dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    model.fit(X_train, y_train)
    mean, lower, upper = model.predict(X_test)

    metrics = compute_all_metrics(mean, lower, upper, y_test, model.alpha)
    metrics["scale_factor"] = np.mean(np.abs(y))
    return metrics 

def save_results(data, regressor_name, path): 
    with open(path / f"{regressor_name}.pkl", 'wb') as f: 
        pickle.dump(data, f)

def run_regressor_test(BASE_SAVE_DIR, model, seed, filename, test_size): 
    DATASET_PATH = Path(__file__).absolute().parent / "datasets"
    datasets = { 
        "concrete": "concrete.xls", 
        #"energy": "energy_efficiency.xlsx", 
        #"kin8nm": "kin8nm.arff", 
        #"naval": "naval_propulsion.txt", 
        #"power": "power_plant.xlsx", 
        #"protein": "protein_structure.csv", 
        #"wine": "winequality-red.csv", 
        #"yacht": "yacht_hydrodynamics.txt"
    }

    regressor_results = {}
    for name, file in datasets.items(): 
        print(f"\n Loading dataset: {name}")
        X, y = load_unformatted_dataset(DATASET_PATH / file)
        metrics = test_regressor(model, X, y, name, seed=seed, test_size=test_size)
        regressor_results[name] = metrics 
        print(metrics)
    print(regressor_results)
    save_results(regressor_results, filename, BASE_SAVE_DIR)

def print_results(path): 
    with open(path, 'rb') as f: 
        results = pickle.load(f)
    for dataset, metrics in results.items(): 
        print(f"{dataset}")
        print("=" * len(dataset))
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"LL: {-metrics['nll_gaussian']:.2f}")
        print(f"Coverage: {metrics['coverage']:.3f}")
        print(f"Average Interval Width: {metrics['average_interval_width']}")
        print(f"Interval Score: {metrics['interval_score']}")
        print(f"CQR Scale Factor: {metrics['scale_factor']}")
        print("=" * len(dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout = MCDropoutRegressor(
        hidden_sizes = [50], 
        dropout = 0.05, 
        use_paper_weight_decay=True,
        alpha = 0.05, 
        n_samples = 100, 
        epochs = 400, 
        batch_size=32,
        learning_rate = 1e-3,
        device=device, 
        use_wandb=False
    )

deep_ens = DeepEnsembleRegressor(
    n_estimators=5, 
    hidden_sizes = [50], 
    n_jobs=2,
    alpha = 0.05, 
    batch_size=100, 
    learning_rate=1e-2, 
    epochs=40, 
    device=device, 
    scale_data=True,
    use_wandb=False
)

cqr = ConformalQuantileRegressor(
    hidden_sizes = [64, 64], 
    cal_size=0.5,
    alpha=0.1,
    dropout=0.1, 
    epochs=1000, 
    batch_size=64, 
    learning_rate=5e-4, 
    optimizer_kwargs = {"weight_decay": 1e-6}, 
    device=device, 
    scale_data=True, 
    use_wandb=False
)

k_fold_cqr = KFoldCQR(
    hidden_sizes = [64, 64], 
    n_estimators=5, 
    n_jobs=1,
    alpha = 0.1, 
    requires_grad=True,
    dropout=0.1, 
    epochs=1000, 
    batch_size=64, 
    learning_rate=5e-4, 
    optimizer_kwargs = {"weight_decay": 1e-6}, 
    device="cpu", 
    scale_data=True, 
    use_wandb=False
)

conformal_ens = ConformalEnsRegressor(
    hidden_sizes=[64, 64], 
    n_estimators=5, 
    cal_size=0.5,
    n_jobs = 2, 
    alpha=0.1, 
    dropout=0.1, 
    pred_with_dropout=False,
    epochs=1000, 
    batch_size=64, 
    learning_rate=5e-4, 
    optimizer_kwargs = {"weight_decay": 1e-6}, 
    device="cpu", 
    scale_data=True, 
    use_wandb=False
)

if __name__ == "__main__": 
    torch.autograd.set_detect_anomaly(True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    BASE_SAVE_DIR = Path.home() / ".uqregressors" / "validation"
    BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    seed = 42

    models = {
        "deep_ens": (deep_ens, 0.05),
        "dropout": (dropout, 0.1),
        "cqr": (cqr, 0.2),
        "conformal_ens": (conformal_ens, 0.2),
        "k_fold_cqr": (k_fold_cqr, 0.2),
    }

    for name, (model, test_size) in models.items(): 
       run_regressor_test(BASE_SAVE_DIR, model, seed, name, test_size=test_size)
       
    print_results(BASE_SAVE_DIR / "cqr.pkl")
    print_results(BASE_SAVE_DIR / "k_fold_cqr.pkl")
    print_results(BASE_SAVE_DIR / "conformal_ens.pkl")
    print_results(BASE_SAVE_DIR / "deep_ens.pkl")
