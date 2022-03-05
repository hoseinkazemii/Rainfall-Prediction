from imblearn.over_sampling import SMOTE

def oversample(X_train, y_train, **params):
	K_SMOTE = params.get("K_SMOTE")
	verbose = params.get("verbose")

	if verbose:
		print("oversampling minority class...")

	oversampler = SMOTE(sampling_strategy = 'minority', k_neighbors = 5, n_jobs = 6)

	X_train, y_train = oversampler.fit_resample(X_train, y_train)

	return X_train, y_train