def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tp=fp=tn=fn=0
#     print('prediction:\n', prediction)
#     print()
#     print('ground_truth:\n', ground_truth)
    
    for i in range(len(prediction)):
        if (prediction[i] == True) and (ground_truth[i] == True):
            tp+=1
        if (prediction[i] == True) and (ground_truth[i] == False):
            fp+=1
        if (prediction[i] == False) and (ground_truth[i] == False):
            tn+=1
        if (prediction[i] == False) and (ground_truth[i] == True):
            fn+=1
            
#     accuracy = 100.0 * np.sum(np.inclose(prediction, ground_truth))/float(len(prediction))

    accuracy = (tp + tn)/(float(len(prediction)))
    
    try:
        precision = tp/ (tp+fp)
    except:
        pass
    
    try:
        recall = tp / (tp+fn)
    except:
        pass

    try:
        f1 = 2 * tp / (2 * tp + fp + fn)
    except:
        pass

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    
    '''
    # TODO: Implement computing accuracy
    
    count = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            count+=1
    
    accuracy = count / (float(len(prediction)))
  
    return accuracy
#     pass