def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    try:
        count = 0
        for i in range(len(prediction)):
            if prediction[i] == ground_truth[i]:
                count+=1

        accuracy = count / (float(len(prediction)))

    except:
        raise Exception("Not implemented!")
    
    return accuracy

    return 0
