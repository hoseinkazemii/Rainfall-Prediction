from ._cyclic_datetime import _cyclic_datetime

import pandas as pd

def make_datetime_cols(df, **params):
	verbose = params.get("verbose")

	if verbose:
		print("making datetime columns...")

	df['Date']= pd.to_datetime(df['Date'])

	df['Day'] = df.Date.dt.day
	df['Month'] = df.Date.dt.month
	df['Year'] = df.Date.dt.year

	df = _cyclic_datetime(df, 'Day', 31)
	df = _cyclic_datetime(df, 'Month', 12)

	return df