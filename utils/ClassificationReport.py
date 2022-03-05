#Loading dependencies
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def evaluate_classification(*args, **params):
    '''Report classification results
    
    *args should be lists of [label,
                              X_train,
                              y_true (y_train or y_test),
                              y_pred (y_pred_train or y_pred_test)]
    '''

    model_name = params.get('model_name')
    logger = params.get('logger')

    for ls in args:
        label, x, y_true, y_pred = ls

        logger.info(f"----------Classification Report for {model_name}-{label}------------\n" + \
                        str(classification_report(y_true, y_pred))+"\n")
        logger.info(f"----------Confusion Matrix for {model_name}-{label}------------\n" + \
                        str(confusion_matrix(y_true, y_pred))+"\n")
        logger.info(f'----------Accurcay for {label}------------\n' + \
                        str(round(accuracy_score(y_true, y_pred),4)))
        
        print (classification_report(y_true, y_pred))
        print (f'Accuracy score for {model_name}-{label}', round(accuracy_score(y_true, y_pred),4))
        print ("------------------------------------------------")

        mse_ = MSE(y_true, y_pred)
        mae_ = MAE(y_true, y_pred)
        
        # Reporting the quantitative results
        report_str = f"{label}, "\
                        f"RMSE={mse_**0.5:.4f}, "\
                                f"MAE={mae_:.4f}"

        logger.info(report_str)