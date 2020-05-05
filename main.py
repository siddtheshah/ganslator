import tensorflow as tf
import numpy as np
import network.combined_model as cm
import network.dataset_util as ds_util

import os
import json
import argparse

parser = argparse.ArgumentParser(description='Arguments for training/eval.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--model_name', help='Specify a run name. (Required)', action='store')
parser.add_argument('--overwrite', default=False, help='Whether to overwrite existing models with the same name.',
                    action='store')
parser.add_argument('--dataset', default='ravdess', help='Which dataset to use')
args = parser.parse_args()


def get_dataset_from_args(configs):
    if args.dataset == 'ravdess':
        if not os.path.isdir(os.path.join(configs['data_dir'], 'ravdess')):
            raise FileNotFoundError("Could not find ravdess path. Aborted")
        return ds_util.create_dataset_from_ravdess(configs['data_dir'])



def run_training(configs, model_name):
    model = cm.GANslator(r_scale=configs['r_scale'], feature_size=configs['mel_bins'], filter_dim=configs['filter_dim'])

    model.save_to_path(os.path.join(configs["storage_dir"], model_name))


def train_model(configs, model_name):
    if configs['use_gpu']:
        with tf.device('/GPU:0'):
            run_training(configs, model_name)
    else:
        run_training(configs, model_name)

    print("Model finished training.")


def eval_model(configs, model_name):
    model = cm.GANslator(r_scale=configs['r_scale'], feature_size=configs['mel_bins'], filter_dim=configs['filter_dim'])
    model.load_from_path(os.path.join(configs["storage_dir"], model_name))


def main():
    if not args.model_name:
        print("No model name specified. Aborted.")
        return

    with open('config.json') as config_file:
        configs = json.load(config_file)

        if os.path.isdir(os.path.join(configs["storage_dir"], args.model_name)) and not args.overwrite:
            raise FileExistsError("There's an existing model with the same name. Specify --overwrite. Aborted.")

        if args.train:
            train_model(configs, args.model_name)

        if args.eval:
            eval_model(configs, args.model_name)


if __name__ == "__main__":
    main()
