import logging

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

from util.logginghandler import TQDMLoggingHandler

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[TQDMLoggingHandler()])


def evaluatation_scores(test, target_label, prediction_label):
    confusion_matrix_values = confusion_matrix(test[prediction_label], test[target_label]).ravel()
    accuracy = accuracy_score(test[prediction_label], test[target_label])
    macro_f1 = f1_score(test[target_label], test[prediction_label], average='macro')
    weighted_f1 = f1_score(test[target_label], test[prediction_label], average='weighted')
    weighted_recall = recall_score(test[target_label], test[prediction_label], average='weighted')
    weighted_precision = precision_score(test[target_label], test[prediction_label], average='weighted')

    return confusion_matrix_values, accuracy, weighted_f1, macro_f1, weighted_recall, weighted_precision


def print_model(model):
    logging.info('*****************************************')
    total = 0
    for name, w in model.named_parameters():
        total += w.nelement()
        logging.info('{} : {}  {} parameters'.format(name, w.shape, w.nelement()))
    logging.info('Total {} parameters'.format(total))
    logging.info('*****************************************')


def draw_graph(n_epohs, valid_losses, trained_losses, path):
    epoh = list(range(1, n_epohs + 1))

    df = pd.DataFrame(
        {'epoh': epoh,
         'validation_loss': valid_losses,
         'training_losses': trained_losses
         })

    sns.set_style("whitegrid")
    ax = sns.lineplot(x="epoh", y="value", hue='variable', data=pd.melt(df, ['epoh']))
    fig = ax.get_figure()
    fig.savefig(path)
    fig.clf()
