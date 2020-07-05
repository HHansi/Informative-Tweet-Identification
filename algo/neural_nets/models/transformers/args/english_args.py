from multiprocessing import cpu_count

TEMP_DIRECTORY = "temp/data"
TRAIN_FILE = "train.tsv"
TEST_FILE = "test.tsv"
DEV_RESULT_FILE = "dev_result.tsv"
DEV_EVAL_FILE = 'dev_eval.txt'
RESULT_FILE = "result.csv"
SUBMISSION_FOLDER = "transformers"
SUBMISSION_FILE = "transformers"
MODEL_TYPE = "bert"
MODEL_NAME = "digitalepidemiologylab/covid-twitter-bert"
HASOC_TRANSFER_LEARNING = False
USE_DISTANT_LEARNING = False

english_args = {
    'output_dir': 'temp/outputs/',
    "best_model_dir": "temp/outputs/best_model",
    'cache_dir': 'temp/cache_dir/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 3,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,
    'n_fold': 5,

    'logging_steps': 80,
    'save_steps': 80,
    "no_cache": False,
    'save_model_every_epoch': True,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 80,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
}
