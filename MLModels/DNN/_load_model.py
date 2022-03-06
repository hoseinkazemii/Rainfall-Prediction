from tensorflow.keras.models import load_model

def _load_model(*args, **kwargs):
	
	should_checkpoint = kwargs.get('should_checkpoint')
	DNN_model_directory = kwargs.get('DNN_model_directory')

	# load json and create model
	if should_checkpoint:
		model_type = 'BestModel'
	else:
		model_type = 'SavedModel'

	model = load_model(DNN_model_directory + "/" +  f"{model_type}.h5")

	return model
