from ._encode_rain_cols import _encode_rain_cols

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def cat_dummies(df, **params):
	verbose = params.get("verbose")
	CB = params.get("CB")

	if verbose:
		print("making dummy variables for categorical columns...")

	if not CB:
		df.drop(columns = ["Date", "Day", "Month", "Year"], inplace = True)

		df = _encode_rain_cols(df, **params)

		categorical = [col for col in df.columns if df[col].dtypes == 'O']

		df = pd.get_dummies(df, columns = categorical, drop_first = False)
	else:
		df.drop(columns = ["Date", "Day", "Month", "Year"], inplace = True)

		df = _encode_rain_cols(df, **params)

		numerical = [col for col in df.columns if df[col].dtypes != 'O']

		x_scaler = MinMaxScaler(feature_range = (0,1))
		numeric_transformer = Pipeline(steps=[('scaler', x_scaler)])

		preprocessor = ColumnTransformer(
		    transformers=[
		        ('num', numeric_transformer, numerical)])


	return df


