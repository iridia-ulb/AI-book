def print_metrics(labels, predictions):
    """Calcule les métriques
    Source: https://github.com/iridia-ulb/AI-book
    """
    # True Positive, True Negative, False Positive, False Negative
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 'spam' and predictions[i] == 'spam')
        true_neg += int(labels[i] == 'ham' and predictions[i] == 'ham')
        false_pos += int(labels[i] == 'ham' and predictions[i] == 'spam')
        false_neg += int(labels[i] == 'spam' and predictions[i] == 'ham')
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
    Fscore = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    accuracy = (true_pos + true_neg) / (
        true_pos + true_neg + false_pos + false_neg
    ) if true_pos + true_neg + false_pos + false_neg != 0 else 0

    print("Metrics")
    print("  Precision: ", precision)
    print("  Recall: ", recall)
    print("  F-score: ", Fscore)
    print("  Accuracy: ", accuracy)
    return((precision,recall,Fscore,accuracy))
