# Created by Hansi at 8/14/2020
import os

import numpy as np
import pandas as pd

from algo.neural_nets.models.transformers.common.data_converter import encode, decode


def majority_class_ensemble(predictions):
    # select majority class of each instance (row)
    final_predictions = []
    for row in predictions:
        row = row.tolist()
        final_predictions.append(int(max(set(row), key=row.count)))
    return final_predictions


# calculated weighted average for final prediction
# note - round 0.5 to 1
def weighted_ensemble(predictions, weights):
    final_predictions = []
    for row in predictions:
        weighted_preds = [a * b for a, b in zip(row, weights)]
        weighted_avg = sum(weighted_preds) / len(weighted_preds)
        if weighted_avg == 0.5:
            final_predictions.append(1)
        else:
            final_predictions.append(round(weighted_avg))
    return final_predictions


def ensemble_results(folder_path, output_file_path, weights=None):
    # intialise results size
    instances = 0
    models = 0
    for root, dirs, files in os.walk(folder_path):
        models = len(files)
        for file in files:
            result = pd.read_csv(os.path.join(folder_path, file), sep='\t')
            instances = len(result)
            break
        break
    predictions = np.zeros((instances, models))

    # read model predictions
    for root, dirs, files in os.walk(folder_path):
        i = 0
        for file in files:
            result = pd.read_csv(os.path.join(folder_path, file), sep='\t')
            preds = encode(result['predictions'])
            predictions[:, i] = preds
            i += 1

    final_predictions = majority_class_ensemble(predictions)

    # decode final predictions
    final_predictions = decode(final_predictions)

    output_file = open(output_file_path, 'w', encoding='utf-8')
    for pred in final_predictions:
        output_file.write(pred + '\n')
    output_file.close()


if __name__ == "__main__":
    folder_path = "../results/ensemble/"
    output_file_path = "../results/ensemble/output.txt"
    ensemble_results(folder_path, output_file_path)
