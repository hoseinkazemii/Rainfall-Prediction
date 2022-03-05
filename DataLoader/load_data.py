import pandas as pd

def load_data(**params):
	data_path = params.get("data_path")
	verbose = params.get("verbose")

	if verbose:
		print("loading Weather data...")

	df = pd.read_csv(data_path)

	return df