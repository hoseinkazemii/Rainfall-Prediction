from ._encode_rain_cols import _encode_rain_cols

import pandas as pd

def cat_dummies(df, **params):
	verbose = params.get("verbose")

	if verbose:
		print("making dummy variables for categorical columns...")

	df.drop(columns = ["Date", "Day", "Month", "Year"], inplace = True)

	df = _encode_rain_cols(df, **params)

	categorical = [col for col in df.columns if df[col].dtypes == 'O']

	df = pd.get_dummies(df, columns = categorical, drop_first = False)

	return df