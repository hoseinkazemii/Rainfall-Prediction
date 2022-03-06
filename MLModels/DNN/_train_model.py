from ._get_data import get_data
from ._get_call_backs import _get_call_backs
from ._save_model import _save_model

from utils import split_and_normalize_data
from utils import evaluate_classification

from utils import Logger

def train_model(X_train, X_test, y_train, y_test, **params):
	
	warm_up = params.get('warm_up')
	starting_batch = params.get('starting_batch')
	n_rounds = params.get('n_rounds')
	split_size = params.get('split_size')
	epochs = params.get('epochs')
	batch_size = params.get('batch_size')
	log = params.get("log")
	model = params.get('model')
	verbose = params.get('verbose')	

	call_back_list = _get_call_backs()

	if verbose:
		print ("Trying to fit to the data...")
	
	model.fit(X_train, y_train,
			  validation_split=split_size,
			  epochs=epochs,
			  batch_size=batch_size,
		      verbose=2, 
		      shuffle=True, 
			  callbacks=call_back_list)

	# Evaluate the model
	train_scores = model.evaluate(X_train, y_train, verbose=2)
	test_scores = model.evaluate(X_test, y_test, verbose=2)
	
	if verbose:
		print (f'Trian_err: {train_scores}, Test_err: {test_scores}')

	log.info(f'batch_number:{batch_number}, Trian_err: {train_scores}, Test_err: {test_scores}')

	_save_model()

	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	evaluate_classification(
		[f'OnTrain', X_train, y_train, y_pred_train],
		[f'OnTest', X_test, y_test, y_pred_test],
		model = model,
		model_name = model_name,
		logger = log)