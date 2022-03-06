import warnings
warnings.filterwarnings("ignore")

from DataLoader import *
from Preprocessing import *
from MLModels import *

def run():
	settings = {
	"data_path" : "./Data/Weather.csv",
	"box_plot_cols" : ['Rainfall','WindSpeed9am','WindSpeed3pm'],
	"verbose" : True,
	"verbose_outliers" : True,
	"imputer" : "Iter_BR",
	"K_SMOTE" : 5,





	}


	#Step1-Preprocessing:

	df = load_data(**settings)
	df = make_datetime_cols(df, **settings)
	# describe_numerical_cols(df, **settings)
	# box_plot(df, **settings)
	# kde_plot(df, **settings)
	df = deal_with_outliers(df, **settings)
	df = deal_with_nulls(df, **settings)
	df = cat_dummies(df, **settings)
	X_train, X_test, y_train, y_test = split_data(df, **settings)
	X_train, X_test = scaler(X_train, X_test, **settings)
	X_train, y_train = oversample(X_train, y_train, **settings)

	#Step2-Training:
	# 2-1: CatBoost
	# cb_settings = {'iterations' : 2,
	# 				'learning_rate' : 0.1,
	# 				'depth' : 9,
	# 				'l2_leaf_reg' : 0.001,
	# 				'loss_function' : 'Logloss',
	# 				# 'loss_function' : 'RMSE',
	# 				'allow_writing_files' : False,
	# 				# 'eval_metric' : "RMSE",
	# 				'eval_metric' : "Accuracy",
	# 				'task_type' : 'CPU',
	# 				'verbose_cb' : 400,
	# 				'boosting_type' : 'Ordered',
	# 				'thread_count' : -1,
	# 				"model_name" : "CatBoost"}


	# myCatBoostModel = CatBoostModel(**{**cb_settings,
	# 	                                          **settings})
	# myCatBoostModel._construct_model()
	# myCatBoostModel.run(X_train, X_test, y_train, y_test)

	# 2-2: RandomForest
	rf_settings = {'n_estimators' : 100,
					'max_depth' : 200,
					'min_samples_split' : 2,
					'min_samples_leaf' : 1,
					'max_features' : 'auto',
					'should_cross_val' : False,
					'n_jobs' : -1,
					"verbose_rf" : 2,
					"model_name" : "RF"}

	myRFModel = RFModel(**{**rf_settings,
										**settings})
	myRFModel._construct_model()
	myRFModel.run(X_train, X_test, y_train, y_test)

	# 2-3: DNN
	DNN_settings = {'DNN_model_directory' : './SavedModels'
			  'layers' : [10,30,20],
			  'input_activation_func' : 'tanh',
			  'hidden_activation_func' : 'relu',
			  'final_activation_func' : 'sigmoid',
			  'loss_func' : 'binary_crossentropy',
			  'epochs' : 10,
			  'min_delta' : 0.00001,
			  'patience' : 10,
		      'batch_size' : 32,
			  'should_early_stop' : False,
			  'should_checkpoint' : False,
		      'regul_type' : 'l2',
			  'act_regul_type' : 'l1',
			  'reg_param' : 0.01,
			  'dropout' : 0.2,
			  'optimizer' : 'adam',
			  'random_state' : 42,
			  'output_dim' : 1,
			  'warm_up' : False,
			  'model_name' : 'DNN',}

	myDNNModel = DNNModel(**{**DNN_settings,
											**settings})
	myDNNLeakDetector._construct_model()
	myDNNLeakDetector.run(X_train, X_test, y_train, y_test)















if __name__ == "__main__":
	run()
