from sklearn.model_selection import train_test_split

def split_data_one_climate(df_climate, **params):

	random_state = params.get("random_state")
	verbose = params.get("verbose")

	if verbose:
		print("splitting data...")

	X = df_climate.drop('RainTomorrow', axis=1)
	y = df_climate['RainTomorrow']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
														random_state = random_state, shuffle = False)

	return X_train, X_test, y_train, y_test