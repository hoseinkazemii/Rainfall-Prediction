#Loading dependencies
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def evaluate_classification(*args, **params):
    '''Report classification results
    
    *args should be lists of [label,x , y_true, y_pred]
    '''

    direc = params.get('direc')
    model = params.get('model', None)
    model_name = params.get('model_name')
    logger = params.get('logger')
    slicer = params.get('slicer', 1)

    for ls in args:
        label, x, y_true, y_pred = ls

        print(classification_report(y_true, y_pred))
        raise ValueError

        logger.info(f"----------Classification Report for {model_name}-{label}------------\n" + \
                        str(classification_report(y_true, y_pred))+"\n")
        logger.info(f"----------Confusion Matrix for {model_name}-{label}------------\n" + \
                        str(confusion_matrix(y_true, y_pred))+"\n")
        logger.info(f'----------Accurcay for {label}------------\n' + \
                        str(round(accuracy_score(y_true, y_pred),4)))
        
        print (classification_report(y_true, y_pred))
        print (f'Accuracy score for {model_name}-{label}', round(accuracy_score(y_true, y_pred),4))
        print ("------------------------------------------------")    
