


def split_data_all_climates(df_all, df_climate, **params):
	verbose = params.get("verbose")

	if verbose:
		print("splitting data...")

	X_train = df_all.drop(columns=['RainTomorrow'], axis=1)
	X_test = df_climate.drop(columns=['RainTomorrow'], axis=1)
	y_train = df_all['RainTomorrow']
	y_test = df_climate['RainTomorrow']

	return X_train, X_test, y_train, y_test