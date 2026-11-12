"""foresurgery — counterfactual forecast surgery on issued rolling forecasts.

The package implements the experiment of the accompanying paper: take the
load/PV forecasts that Schneider Electric actually issued to 70 industrial
microgrids, replace ("heal") one horizon band of the issued forecast with the
realised values, re-solve the battery dispatch LP, and measure how much of the
forecast-induced cost regret that band is responsible for.

Modules
-------
config       directories, tariff, battery defaults, horizon bands
data         EMSx warehouse construction and site loading
dispatch     battery dispatch LP and open-loop execution
surgery      the heal/degrade operator and the experiment campaign
clusters     load-shape site typing (k-means)
attribution  cost-weighted band attribution with bootstrap intervals
corrector    feature matrix and rolling-archive splits for retraining
plotting     Nature/NPG figure style used in the paper
"""

__version__ = "1.0.0"

__all__ = ["config", "data", "dispatch", "surgery", "clusters",
           "attribution", "corrector", "plotting"]
