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
	"random_state" : 42,
	"imputer" : "Iter_BR",
	"K_SMOTE" : 5,
	"Koppen_climate" : 'BSh',
	# ['Cfa', 'BSh', 'Cfb', 'BSk',
	#  'Csb', 'Am', 'Aw', 'BWh', 'Csa']





	}

	#Step1: Training models on "ALL" climates excluding one,
	#		Testing model on the excluded one.
	#Step1-1-Preprocessing:

	df = load_data(**settings)
	df = make_datetime_cols(df, **settings)
	# describe_numerical_cols(df, **settings)
	# box_plot(df, **settings)
	# kde_plot(df, **settings)
	df = deal_with_outliers(df, **settings)
	df = deal_with_nulls(df, **settings)
	df = cat_dummies(df, **settings)
	df_climate, df_all = split_climate(df, **settings)	
	X_train, X_test, y_train, y_test = split_data_all_climates(df_all, df_climate, **settings)
	X_train, X_test = scaler(X_train, X_test, **settings)
	# X_train, y_train = oversample(X_train, y_train, **settings)

	#Step1-2-Training:
	# 1-2-1: CatBoost

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
	# 				"model_name" : "CatBoost",
	# 				"approach" : "AllClimates",}


	# myCatBoostModel = CatBoostModel(**{**cb_settings,
	# 	                                          **settings})
	# myCatBoostModel._construct_model()
	# myCatBoostModel.run(X_train, X_test, y_train, y_test)

	# 1-2-2: RandomForest

	# rf_settings = {'n_estimators' : 100,
	# 				'max_depth' : 200,
	# 				'min_samples_split' : 2,
	# 				'min_samples_leaf' : 1,
	# 				'max_features' : 'auto',
	# 				'should_cross_val' : False,
	# 				'n_jobs' : -1,
	# 				"verbose_rf" : 2,
	# 				"model_name" : "RF",
	# 				"approach" : "AllClimates",}

	# myRFModel = RFModel(**{**rf_settings,
	# 									**settings})
	# myRFModel._construct_model()
	# myRFModel.run(X_train, X_test, y_train, y_test)

	# 1-2-3: DNN

	# DNN_settings = {'DNN_model_directory' : './SavedModels/DNN',
	# 		  'layers' : [10,30,20],
	# 		  'input_activation_func' : 'tanh',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'final_activation_func' : 'sigmoid',
	# 		  'loss_func' : 'binary_crossentropy',
	# 		  'epochs' : 2,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 32,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.2,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 42,
	# 		  'split_size' : 0.2,
	# 		  'output_dim' : 1,
	# 		  'warm_up' : False,
	# 		  'model_name' : 'DNN',
	# 		  "approach" : "AllClimates",}

	# myDNNModel = DNNModel(**{**DNN_settings,
	# 										**settings})
	# myDNNModel._construct_model(df_all)
	# myDNNModel.run(X_train, X_test, y_train, y_test)

	# 1-2-4: LSTM

	# LSTM_settings = {'LSTM_model_directory' : './SavedModels',
	# 		  'layers' : [15,30,35],
	# 		  'input_activation_func' : 'relu',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'final_activation_func' : 'sigmoid',
	# 		  'loss_func' : 'binary_crossentropy',
	# 		  'epochs' : 50,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 16,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.2,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 42,
	# 		  'split_size' : 0.2,
	# 		  'output_dim' : 1,
	# 		  'warm_up' : False,
	# 		  'model_name' : 'LSTM',
	# 		  "approach" : "AllClimates",}

	# myLSTMModel = LSTMModel(**{**LSTM_settings,
	# 										**settings})
	# X_train, X_test = reshape_X(X_train, X_test, **settings)
	# myLSTMModel._construct_model(df_all, X_train)
	# myLSTMModel.run(X_train, X_test, y_train, y_test)




	#Step2: Training models on "ONE" climates excluding one,
	#		Testing model on the excluded one.
	#Step2-1-Preprocessing:

	# df = load_data(**settings)
	# df = make_datetime_cols(df, **settings)
	# # describe_numerical_cols(df, **settings)
	# # box_plot(df, **settings)
	# # kde_plot(df, **settings)
	# df = deal_with_outliers(df, **settings)
	# df = deal_with_nulls(df, **settings)
	# df = cat_dummies(df, **settings)
	# df_climate, _ = split_climate(df, **settings)	
	# X_train, X_test, y_train, y_test = split_data_one_climate(df_climate, **settings)
	# X_train, X_test = scaler(X_train, X_test, **settings)
	# # X_train, y_train = oversample(X_train, y_train, **settings)

	# Step2-2-Training:
	# 2-2-1: CatBoost

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
	# 				"model_name" : "CatBoost",
	# 				"approach" : "OneClimate",}


	# myCatBoostModel = CatBoostModel(**{**cb_settings,
	# 	                                          **settings})
	# myCatBoostModel._construct_model()
	# myCatBoostModel.run(X_train, X_test, y_train, y_test)

	# 2-2-2: RandomForest

	# rf_settings = {'n_estimators' : 100,
	# 				'max_depth' : 200,
	# 				'min_samples_split' : 2,
	# 				'min_samples_leaf' : 1,
	# 				'max_features' : 'auto',
	# 				'should_cross_val' : False,
	# 				'n_jobs' : -1,
	# 				"verbose_rf" : 2,
	# 				"model_name" : "RF",
	# 				"approach" : "OneClimate",}

	# myRFModel = RFModel(**{**rf_settings,
	# 									**settings})
	# myRFModel._construct_model()
	# myRFModel.run(X_train, X_test, y_train, y_test)

	# 2-2-3: DNN

	# DNN_settings = {'DNN_model_directory' : './SavedModels',
	# 		  'layers' : [10,30,20],
	# 		  'input_activation_func' : 'tanh',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'final_activation_func' : 'sigmoid',
	# 		  'loss_func' : 'binary_crossentropy',
	# 		  'epochs' : 2,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 32,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.2,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 42,
	# 		  'split_size' : 0.2,
	# 		  'output_dim' : 1,
	# 		  'warm_up' : False,
	# 		  'model_name' : 'DNN',
	# 		  "approach" : "OneClimate",}

	# myDNNModel = DNNModel(**{**DNN_settings,
	# 										**settings})
	# myDNNModel._construct_model(df_climate)
	# myDNNModel.run(X_train, X_test, y_train, y_test)

	# 1-2-4: LSTM

	# LSTM_settings = {'LSTM_model_directory' : './SavedModels',
	# 		  'layers' : [10,20],
	# 		  'input_activation_func' : 'relu',
	# 		  'hidden_activation_func' : 'relu',
	# 		  'final_activation_func' : 'sigmoid',
	# 		  'loss_func' : 'binary_crossentropy',
	# 		  'epochs' : 30,
	# 		  'min_delta' : 0.00001,
	# 		  'patience' : 10,
	# 	      'batch_size' : 64,
	# 		  'should_early_stop' : False,
	# 		  'should_checkpoint' : False,
	# 	      'regul_type' : 'l2',
	# 		  'act_regul_type' : 'l1',
	# 		  'reg_param' : 0.01,
	# 		  'dropout' : 0.2,
	# 		  'optimizer' : 'adam',
	# 		  'random_state' : 42,
	# 		  'split_size' : 0.2,
	# 		  'output_dim' : 1,
	# 		  'warm_up' : False,
	# 		  'model_name' : 'LSTM',
	# 		  "approach" : "OneClimate",}

	# myLSTMModel = LSTMModel(**{**LSTM_settings,
	# 										**settings})
	# X_train, X_test = reshape_X(X_train, X_test, **settings)
	# myLSTMModel._construct_model(df_climate, X_train)
	# myLSTMModel.run(X_train, X_test, y_train, y_test)













if __name__ == "__main__":
	run()
