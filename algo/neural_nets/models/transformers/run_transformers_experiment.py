import logging
import os
import shutil

import pandas as pd
import sklearn
import torch
import numpy as np

from algo.neural_nets.common.utility import evaluatation_scores, save_eval_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from algo.neural_nets.models.transformers.common.data_converter import encode, decode
from algo.neural_nets.models.transformers.common.evaluation import f1, labels, pos_label
from util.logginghandler import TQDMLoggingHandler

from algo.neural_nets.common.preprocessor import transformer_pipeline
from algo.neural_nets.models.transformers.args.english_args import TEMP_DIRECTORY, RESULT_FILE, MODEL_TYPE, MODEL_NAME, \
    english_args, HASOC_TRANSFER_LEARNING, USE_DISTANT_LEARNING, DEV_RESULT_FILE, SUBMISSION_FOLDER, SUBMISSION_FILE, \
    DEV_EVAL_FILE
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
from project_config import SEED, TRAINING_DATA_PATH, VALIDATION_DATA_PATH, CONFUSION_MATRIX, F1, RECALL, PRECISION, \
    ACCURACY

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])

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

# Create a ClassificationModel
if HASOC_TRANSFER_LEARNING:
    print()
# logging.info("Started HASOC Transfer Learning")
# model_dir = run_hasoc_experiment()
# model = ClassificationModel(MODEL_TYPE, model_dir, args=english_args,
#                             use_cuda=torch.cuda.is_available())

elif USE_DISTANT_LEARNING:
    print()
# logging.info("Started Distant Transfer Learning")
# model_dir = run_transfer_learning_experiment()
# model = ClassificationModel(MODEL_TYPE, model_dir, args=english_args,
#                             use_cuda=torch.cuda.is_available())

else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=english_args,
                                use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument

# Train the model
logging.info("Started Training")

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev), english_args["n_fold"]))

if english_args["evaluate_during_training"]:
    for i in range(english_args["n_fold"]):
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
        model.train_model(train, eval_df=eval_df, f1=f1, accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, english_args["best_model_dir"], args=english_args,
                                    use_cuda=torch.cuda.is_available())

        predictions, raw_outputs = model.predict(dev_sentences)
        dev_preds[:, i] = predictions
    # select majority class of each instance (row)
    final_predictions = []
    for row in dev_preds:
        row = row.tolist()
        final_predictions.append(max(set(row), key=row.count))
    dev['predictions'] = final_predictions
else:
    model.train_model(train, f1=f1, accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(dev_sentences)
    dev['predictions'] = predictions

dev['predictions'] = decode(dev['predictions'])

logging.info("Started Evaluation")
results = evaluatation_scores(dev, 'class', 'predictions', labels, pos_label)

logging.info("Confusion Matrix {}".format(results[CONFUSION_MATRIX]))
logging.info("Accuracy {}".format(results[ACCURACY]))
logging.info("F1 {}".format(results[F1]))
logging.info("Recall {}".format(results[RECALL]))
logging.info("Precision {}".format(results[PRECISION]))

dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
save_eval_results(results, os.path.join(TEMP_DIRECTORY, DEV_EVAL_FILE))

logging.info("Finished Evaluation")

# logging.info("Started Testing")
# test_sentences = test['text'].tolist()
#
# if english_args["evaluate_during_training"]:
#     model = ClassificationModel(MODEL_TYPE, english_args["best_model_dir"], args=english_args,
#                                 use_cuda=torch.cuda.is_available())
#
# test_predictions, raw_outputs = model.predict(test_sentences)
#
# test['Label'] = le.inverse_transform(test_predictions)
#
# test = test[['id', 'Label']]
# test.to_csv(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER, RESULT_FILE), header=False, sep=',', index=False,
#             encoding='utf-8')
#
# shutil.make_archive(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'zip',
#                     os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))
#
# logging.info("Finished Testing")


