from sklearn.preprocessing import MinMaxScaler

def scaler(X_train, X_test, **params):
	verbose = params.get("verbose")

	if verbose:
		print("scaling data...")

	x_scaler = MinMaxScaler(feature_range = (0,1))

	X_train = x_scaler.fit_transform(X_train)
	X_test = x_scaler.transform(X_test)

	return X_train, X_test