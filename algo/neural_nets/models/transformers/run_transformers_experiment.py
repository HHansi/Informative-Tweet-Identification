import logging
import os
import shutil
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split

from algo.neural_nets.common.preprocessor import transformer_pipeline
from algo.neural_nets.common.utility import evaluatation_scores, save_eval_results
from algo.neural_nets.models.transformers.args.args import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, \
    args, DEV_RESULT_FILE, SUBMISSION_FOLDER, DEV_EVAL_FILE, SEED, LANGUAGE_FINETUNE, language_modeling_args
from algo.neural_nets.models.transformers.common.data_converter import encode, decode
from algo.neural_nets.models.transformers.common.evaluation import f1, labels, pos_label
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
from algo.neural_nets.models.transformers.language_modeling import LanguageModelingModel
from project_config import TRAINING_DATA_PATH, VALIDATION_DATA_PATH, CONFUSION_MATRIX, F1, RECALL, PRECISION, \
    ACCURACY


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

train = pd.read_csv(TRAINING_DATA_PATH, sep='\t')
dev = pd.read_csv(VALIDATION_DATA_PATH, sep='\t')

# train, dev = train_test_split(full, test_size=0.2, random_state=SEED)
train['class'] = encode(train["Label"])
train['text'] = train["Text"]
train = train[['text', 'class']]
train['text'] = train['text'].apply(lambda x: transformer_pipeline(x))

dev['class'] = encode(dev["Label"])
dev['text'] = dev["Text"]
dev = dev[['text', 'class']]
dev['text'] = dev['text'].apply(lambda x: transformer_pipeline(x))

# test['text'] = test["Label"]
# test['text'] = test['text'].apply(lambda x: transformer_pipeline(x))

if LANGUAGE_FINETUNE:
    train_list = train['text'].tolist()
    dev_list = dev['text'].tolist()
    complete_list = train_list + dev_list
    lm_train = complete_list[0: int(len(complete_list)*0.8)]
    lm_test = complete_list[-int(len(complete_list)*0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_args["best_model_dir"]


# Train the model
print("Started Training")

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev), args["n_fold"]))

if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
        model.train_model(train_df, eval_df=eval_df, f1=f1, accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
                                    use_cuda=torch.cuda.is_available())

        predictions, raw_outputs = model.predict(dev_sentences)
        dev_preds[:, i] = predictions
        print("Completed Fold {}".format(i))
    # select majority class of each instance (row)
    final_predictions = []
    for row in dev_preds:
        row = row.tolist()
        final_predictions.append(int(max(set(row), key=row.count)))
    dev['predictions'] = final_predictions
else:
    model.train_model(train, f1=f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(dev_sentences)
    dev['predictions'] = predictions

dev['predictions'] = decode(dev['predictions'])
dev['class'] = decode(dev['class'])

time.sleep(5)

print("Started Evaluation")
results = evaluatation_scores(dev, 'class', 'predictions', labels, pos_label)

print("Confusion Matrix {}".format(results[CONFUSION_MATRIX]))
print("Accuracy {}".format(results[ACCURACY]))
print("F1 {}".format(results[F1]))
print("Recall {}".format(results[RECALL]))
print("Precision {}".format(results[PRECISION]))

dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
save_eval_results(results, os.path.join(TEMP_DIRECTORY, DEV_EVAL_FILE))

print("Finished Evaluation")

