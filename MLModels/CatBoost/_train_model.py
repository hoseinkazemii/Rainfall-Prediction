from utils import evaluate_classification
import numpy as np

def train_model(X_train, X_test, y_train, y_test, **params):
	
	log = params.get("log")
	model = params.get('model')
	verbose = params.get('verbose')
	model_name = params.get("model_name")

	if verbose:
		print ("Trying to fit to the data...")

	categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

	model.fit(X_train, y_train, cat_features=categorical_features_indices, plot=True)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_classification(
		[f'OnTrain', X_train, y_train, y_pred_train],
		[f'OnTest', X_test, y_test, y_pred_test],
		model = model,
		model_name = model_name,
		logger = log)