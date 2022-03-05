from utils import evaluate_classification
from utils import evaluate_regression

def train_model(X_train, X_test, y_train, y_test,**params):
	
	log = params.get("log")
	model = params.get('model')
	verbose = params.get('verbose')
	direc = params.get("report_directory")

	if verbose:
		print ("Trying to fit to the data...")

	model.fit(X_train, y_train)

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_classification(
		[f'OnTrain', X_train, y_train, y_pred_train],
		[f'OnTest', X_test, y_test, y_pred_test],
		model = model,
		model_name = f"CatBoost",
		logger = log,
		slicer = 1,
		direc = direc)

	# evaluate_regression(
	# 	[f'OnTrain', X_train, y_train, y_pred_train],
	# 	[f'OnTest', X_test, y_test, y_pred_test],
	# 	model = model,
	# 	model_name = f"CatBoost",
	# 	logger = log,
	# 	slicer = 1,
	# 	direc = direc)