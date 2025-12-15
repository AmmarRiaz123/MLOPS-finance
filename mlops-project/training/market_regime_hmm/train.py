from pathlib import Path
import json
import shutil
from datetime import datetime
import logging
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import prepare_features
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "market_regime_hmm.pkl"
ARCHIVED_MODELS = MODELS_DIR / "archived"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST = METRICS_DIR / "latest" / "metrics.json"
METRICS_ARCHIVED = METRICS_DIR / "archived"
PLOTS_DIR = METRICS_DIR / "latest" / "plots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
	if path.exists():
		archived_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.now().strftime("%Y%m%dT%H%M%S")
		dest = archived_dir / f"{ts}_{prefix}"
		shutil.move(str(path), str(dest))

def train(n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
	"""
	Fit GaussianHMM on standardized features and save model + regime outputs.
	"""
	# try importing hmmlearn GaussianHMM, fallback to GaussianMixture if not installed
	use_gmm_fallback = False
	try:
		from hmmlearn.hmm import GaussianHMM  # type: ignore
		hmm_backend = "hmmlearn"
	except Exception:
		from sklearn.mixture import GaussianMixture
		use_gmm_fallback = True
		hmm_backend = "gmm_fallback"

	log = logging.getLogger("market_regime_hmm")
	log.info("Preparing features")
	X, returns, vol_series, dates, scaler = prepare_features()

	# fit HMM
	log.info(f"Fitting {hmm_backend} with {n_components} regimes")
	if not use_gmm_fallback:
		model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, random_state=random_state, n_iter=200)
		model.fit(X.values)
		regimes = model.predict(X.values)
	else:
		# GaussianMixture does not model transitions; use it as a density-based regime assigner
		model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state, max_iter=200)
		model.fit(X.values)
		regimes = model.predict(X.values)
	regime_series = pd.Series(regimes, index=dates.index, name="regime")

	# metrics
	dfm = pd.DataFrame({
		"date": dates.values,
		"regime": regimes,
		"return": returns.values,
		"vol_short": vol_series.values
	}, index=dates.index)

	regime_counts = dfm['regime'].value_counts().sort_index().to_dict()
	avg_return = dfm.groupby('regime')['return'].mean().to_dict()
	avg_vol = dfm.groupby('regime')['vol_short'].mean().to_dict()
	# compute transition matrix: if using hmmlearn use model.transmat_, otherwise estimate empirically
	if not use_gmm_fallback and hasattr(model, "transmat_"):
		trans_mat = model.transmat_.tolist()
	else:
		# empirical transition counts
		n = n_components
		counts = [[0 for _ in range(n)] for __ in range(n)]
		for a, b in zip(regimes[:-1], regimes[1:]):
			counts[int(a)][int(b)] += 1
		# normalize to probabilities with safe division
		trans_mat = []
		for row in counts:
			s = sum(row)
			if s > 0:
				trans_mat.append([r / s for r in row])
			else:
				trans_mat.append([0.0 for _ in row])

	# archive old model and save new
	_archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_MODELS, "market_regime_hmm.pkl")
	LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	# save backend info so evaluation/loading knows which type it is
	joblib.dump({"model": model, "scaler": scaler, "features": X.columns.tolist(), "backend": hmm_backend}, LATEST_MODEL_PATH)
	log.info(f"Saved HMM model: {LATEST_MODEL_PATH.resolve()}")

	# ensure metrics dir
	_archive_if_exists(METRICS_LATEST, METRICS_ARCHIVED, "metrics.json")
	METRICS_LATEST.parent.mkdir(parents=True, exist_ok=True)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)

	metrics = {
		"trained_at": datetime.now().isoformat(),
		"n_components": n_components,
		"regime_counts": regime_counts,
		"avg_return_per_regime": {int(k): float(v) for k,v in avg_return.items()},
		"avg_vol_per_regime": {int(k): float(v) for k,v in avg_vol.items()},
		"transition_matrix": trans_mat,
		"hmm_backend": hmm_backend
	}
	with open(METRICS_LATEST, "w") as f:
		json.dump(metrics, f, indent=2)
	log.info(f"Saved metrics: {METRICS_LATEST.resolve()}")

	# save regime sequence CSV
	(dfm.reset_index()[['date','regime']].to_csv(METRICS_DIR / "latest" / "regimes.csv", index=False))

	# plots: price colored by regime (optional)
	try:
		price = pd.Series(np.exp(returns.cumsum()), index=dates.index) * 1  # approximate price path for visualization
		fig, ax = plt.subplots(figsize=(12,4))
		for r in sorted(dfm['regime'].unique()):
			mask = dfm['regime']==r
			ax.plot(dfm.index[mask], price[mask], marker='.', linestyle='-', label=f"regime_{r}")
		ax.set_title("Price segments by regime (approx)")
		ax.legend()
		fig.savefig(PLOTS_DIR / "price_by_regime.png", dpi=150)
		plt.close(fig)
	except Exception as _e:
		log.warning(f"Price-by-regime plot failed: {_e}")

	# volatility per regime
	try:
		fig, ax = plt.subplots(figsize=(8,4))
		regs = sorted(dfm['regime'].unique())
		ax.bar([str(r) for r in regs], [dfm.loc[dfm['regime']==r,'vol_short'].mean() for r in regs])
		ax.set_title("Avg short volatility per regime")
		fig.savefig(PLOTS_DIR / "vol_by_regime.png", dpi=150)
		plt.close(fig)
	except Exception as _e:
		log.warning(f"Vol-by-regime plot failed: {_e}")

	return metrics

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_components", type=int, default=3)
	parser.add_argument("--covariance_type", type=str, default="full")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	res = train(n_components=args.n_components, covariance_type=args.covariance_type, random_state=args.seed)
	print("Training metrics:", res)
