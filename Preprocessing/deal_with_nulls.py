from ._fill_missing_seasonal_num_values import _fill_missing_seasonal_num_values
from ._imputer import _imputer
from ._cat_nulls import _cat_nulls

import pandas as pd

def deal_with_nulls(df, **params):
	verbose = params.get("verbose")

	if verbose:
		print("dealing with null values...")

	locations = df['Location'].unique()

	cols = ['Temp9am', 'Temp3pm','MinTemp','MaxTemp']

	for col in cols:
	    dfs = []

	    for location in locations:
	        dfs.append(_fill_missing_seasonal_num_values(df, location, col))

	    df = pd.concat(dfs)

	cols = ["Evaporation", "Sunshine", "WindGustSpeed", 
			"Humidity9am", "Humidity3pm", "Pressure9am",
			"Pressure3pm", "Cloud9am", "Cloud3pm",]
	
	for col in cols:
		df = _imputer(df, col, **params)

	df = _cat_nulls(df, **params)

	return df
