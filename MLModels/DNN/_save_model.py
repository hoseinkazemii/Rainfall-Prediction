def _save_model(*args, **params):
	
	directory = params.get('directory')
	model_name = params.get("model_name")

	save_address = f"{directory}/" 
	model.save(save_address + f"{model_name}-SavedModel.h5", save_format = 'h5')
	