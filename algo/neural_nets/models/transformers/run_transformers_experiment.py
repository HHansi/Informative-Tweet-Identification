import os
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
    args, DEV_RESULT_FILE, SUBMISSION_FOLDER, DEV_EVAL_FILE, SEED, LANGUAGE_FINETUNE, language_modeling_args, \
    PREPROCESS_TYPE, TEST_RESULT_FILE, SUBMISSION_FILE, INCLUDE_RAW_PREDICTIONS, TAG_RAW, N_CLASSES
from algo.neural_nets.models.transformers.common.data_converter import encode, decode
from algo.neural_nets.models.transformers.common.evaluation import f1, labels, pos_label
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
from algo.neural_nets.models.transformers.language_modeling import LanguageModelingModel
from project_config import TRAINING_DATA_PATH, VALIDATION_DATA_PATH, CONFUSION_MATRIX, F1, RECALL, PRECISION, \
    ACCURACY, TEST_DATA_PATH

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))


# raw preds - [preds_class0, preds_class1, ... preds_classn]
def average_predictions(raw_preds):
    n = len(raw_preds[0])
    avg_preds = np.zeros((n, N_CLASSES))
    for c in range(N_CLASSES):
        preds_c = raw_preds[c]
        avg_vals = [sum(row) / len(row) for row in preds_c]
        avg_preds[:, c] = avg_vals
    final_preds = np.argmax(avg_preds, axis=1)
    return avg_preds, final_preds


def print_results(results):
    print("Confusion Matrix {}".format(results[CONFUSION_MATRIX]))
    print("Accuracy {}".format(results[ACCURACY]))
    print("F1 {}".format(results[F1]))
    print("Recall {}".format(results[RECALL]))
    print("Precision {}".format(results[PRECISION]))


colnames = ['Id', 'Time', 'Text']
train = pd.read_csv(TRAINING_DATA_PATH, sep='\t', names=colnames, header=None)
# dev = pd.read_csv(VALIDATION_DATA_PATH, sep='\t')

# colnames = ['Id', 'Text']
# test = pd.read_csv(TEST_DATA_PATH, sep='\t', names=colnames, header=None)

# train, dev = train_test_split(full, test_size=0.2, random_state=SEED)
# train['class'] = encode(train["Label"])
# train['text'] = train["Text"]
# train = train[['text', 'class']]
# train['text'] = train['text'].apply(lambda x: transformer_pipeline(x, PREPROCESS_TYPE))

# dev['class'] = encode(dev["Label"])
# dev['text'] = dev["Text"]
# dev = dev[['text', 'class']]
# dev['text'] = dev['text'].apply(lambda x: transformer_pipeline(x, PREPROCESS_TYPE))

# test['text'] = test["Text"]
# test = test[['text']]
# test['text'] = test['text'].apply(lambda x: transformer_pipeline(x, PREPROCESS_TYPE))

start_time = time.time()
if LANGUAGE_FINETUNE:
    train_list = train['text'].tolist()
    # dev_list = dev['text'].tolist()
    # complete_list = train_list + dev_list
    complete_list = train_list
    print('train size: ', len(complete_list))

    lm_train = complete_list[0: int(len(complete_list) * 0.8)]
    lm_test = complete_list[-int(len(complete_list) * 0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
    temp_start_time = time.time()
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"),
                      eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    temp_end_time = time.time()
    print('Completed learning in ', int(temp_end_time - temp_start_time), ' seconds')
    MODEL_NAME = language_modeling_args["best_model_dir"]

end_time = time.time()
print('Completed LM in ', int(end_time - start_time), ' seconds')


# Train the model
# print("Started Training")
#
# dev_sentences = dev['text'].tolist()
# dev_preds = np.zeros((len(dev), args["n_fold"]))
# dev_raw_preds = [np.zeros((len(dev), args["n_fold"])) for i in range(N_CLASSES)]
#
# # test_sentences = test['text'].tolist()
# # test_preds = np.zeros((len(test), args["n_fold"]))
# # test_raw_preds = [np.zeros((len(test), args["n_fold"])) for i in range(N_CLASSES)]
#
# if args["evaluate_during_training"]:
#     for i in range(args["n_fold"]):
#         if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
#             shutil.rmtree(args['output_dir'])
#         print("Started Fold {}".format(i))
#         model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
#                                     use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
#         train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
#         model.train_model(train_df, eval_df=eval_df, f1=f1, accuracy=sklearn.metrics.accuracy_score)
#         model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
#                                     use_cuda=torch.cuda.is_available())
#
#         predictions, raw_outputs = model.predict(dev_sentences)
#         dev_preds[:, i] = predictions
#         np_raw_output = np.array(raw_outputs)
#         for j in range(N_CLASSES):
#             dev_raw_preds[j][:, i] = np_raw_output[:, j]
#
#         # test_predictions, test_raw_outputs = model.predict(test_sentences)
#         # test_preds[:, i] = test_predictions
#         # np_test_raw_output = np.array(test_raw_outputs)
#         # for j in range(N_CLASSES):
#         #     test_raw_preds[j][:, i] = np_test_raw_output[:, j]
#
#         print("Completed Fold {}".format(i))
#
#     # select majority class of each instance (row)
#     final_predictions = []
#     for row in dev_preds:
#         row = row.tolist()
#         final_predictions.append(int(max(set(row), key=row.count)))
#     dev['predictions'] = final_predictions
#
#     # final_predictions_test = []
#     # for row in test_preds:
#     #     row = row.tolist()
#     #     final_predictions_test.append(int(max(set(row), key=row.count)))
#     # test['predictions'] = final_predictions_test
#
#     if INCLUDE_RAW_PREDICTIONS:
#         # calculate average of raw prediction
#         avg_raw_preds, final_raw_preds = average_predictions(dev_raw_preds)
#         dev['raw-predictions'] = final_raw_preds
#
#         # avg_raw_preds, final_raw_preds = average_predictions(test_raw_preds)
#         # test['raw-predictions'] = final_raw_preds
#
# else:
#     model.train_model(train, f1=f1, accuracy=sklearn.metrics.accuracy_score)
#     predictions, raw_outputs = model.predict(dev_sentences)
#     dev['predictions'] = predictions
#     # test_predictions, test_raw_outputs = model.predict(test_sentences)
#     # test['predictions'] = test_predictions
#
# dev['predictions'] = decode(dev['predictions'])
# dev['class'] = decode(dev['class'])
#
# # test['predictions'] = decode(test['predictions'])
#
# if INCLUDE_RAW_PREDICTIONS:
#     dev['raw-predictions'] = decode(dev['raw-predictions'])
#     # test['raw-predictions'] = decode(test['raw-predictions'])
#
# time.sleep(5)
#
# print("Started Evaluation")
# results = evaluatation_scores(dev, 'class', 'predictions', labels, pos_label)
# print_results(results)
# save_eval_results(results, os.path.join(TEMP_DIRECTORY, DEV_EVAL_FILE))
#
# if INCLUDE_RAW_PREDICTIONS:
#     print("Evaluation - Raw Outputs")
#     results = evaluatation_scores(dev, 'class', 'raw-predictions', labels, pos_label)
#     print_results(results)
#     save_eval_results(results, os.path.join(TEMP_DIRECTORY, TAG_RAW + "-" + DEV_EVAL_FILE))
#
# dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
# # test.to_csv(os.path.join(TEMP_DIRECTORY, TEST_RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
#
# # output_file = open(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'w', encoding='utf-8')
# # test_preds = test['predictions']
# # for pred in test_preds:
# #     output_file.write(pred + '\n')
# # output_file.close()
# #
# # if INCLUDE_RAW_PREDICTIONS:
# #     output_file = open(os.path.join(TEMP_DIRECTORY, TAG_RAW + "-" + SUBMISSION_FILE), 'w', encoding='utf-8')
# #     test_preds = test['raw-predictions']
# #     for pred in test_preds:
# #         output_file.write(pred + '\n')
# #     output_file.close()
#
# print("Finished Evaluation")
