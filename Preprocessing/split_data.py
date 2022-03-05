from sklearn.model_selection import train_test_split

def split_data(df, **params):

	verbose = params.get("verbose")

	if verbose:
		print("splitting data...")

	X = df.drop('RainTomorrow', axis = 1)
	y = df['RainTomorrow']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
														random_state = 42, shuffle = False)

	return X_train, X_test, y_train, y_test