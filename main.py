import tensorflow as tf
import numpy as np
import network.combined_model as cm
import network.dataset_util as ds_util
import eval.inception

import os
import json
import argparse

parser = argparse.ArgumentParser(description='Arguments for training/eval.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--model_name', help='Specify a run name. (Required)', action='store')
parser.add_argument('--overwrite', default=False, help='Whether to overwrite existing models with the same name.',
                    action='store_true')
parser.add_argument('--dataset', default='ravdess_chunked', help='Which dataset to use')
args = parser.parse_args()

# If using a custom dataset, separate your data into two folders corresponding to different types: "domain1", "domain2"
def get_dataset_from_args(configs):
    if args.dataset == 'ravdess':
        if not os.path.isdir(os.path.join(configs['data_dir'], 'ravdess')):
            raise FileNotFoundError("Could not find ravdess path. Aborted")
        return ds_util.basic_ravdess(os.path.join(configs['data_dir'], 'ravdess'), samples=configs['samples'])
    if args.dataset == 'ravdess_chunked':
        if not os.path.isdir(os.path.join(configs['data_dir'], 'ravdess')):
            raise FileNotFoundError("Could not find ravdess path. Aborted")
        return ds_util.chunked_ravdess(os.path.join(configs['data_dir'], 'ravdess'), chunk_size=configs['samples'], starting_offset=10000)

    input_spec = os.path.join(configs['data_dir'], args.dataset, "domain1", "*")
    output_spec = os.path.join(configs['data_dir'], args.dataset, "domain2", "*")
    return ds_util.create_unconditioned_dataset_from_io_spec(input_spec, output_spec, configs['samples'])


def run_training(configs, model_name, dataset):
    model = cm.GANslator(sample_size=configs['samples'], r_scale=configs['r_scale'], feature_size=configs['mel_bins'],
                         filter_dim=configs['filter_dim'], use_attn=configs['use_attn'])
    model_path = os.path.join(configs["storage_dir"], model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    model.train(dataset, configs['epochs'], configs['batch_size'], configs['save_interval'], configs['result_dir'])
    model.save_to_path(model_path)


def train_model(configs, model_name):
    with tf.device('/CPU:0'):
        dataset = get_dataset_from_args(configs)
    if configs['use_gpu']:
        with tf.device('/GPU:0'):
            run_training(configs, model_name, dataset)
    else:
        with tf.device('/CPU:0'):
            run_training(configs, model_name, dataset)

    print("Model finished training.")


def eval_model(configs, model_name):
    model = cm.GANslator(sample_size=configs['samples'], r_scale=configs['r_scale'], feature_size=configs['mel_bins'],
                         filter_dim=configs['filter_dim'], use_attn=configs['use_attn'])
    model.load_from_path(os.path.join(configs["storage_dir"], model_name))
    dataset = get_dataset_from_args(configs)
    model_results_path = os.path.join(configs["result_dir"], model_name)
    model.sample_for_eval(model_results_path, dataset)
    images_path = os.path.join(model_results_path, "images")
    eval.inception.score(images_path)


def main():
    if not args.model_name:
        print("No model name specified. Aborted.")
        return

    with open('config.json') as config_file:
        configs = json.load(config_file)

        if not os.path.isdir(configs["storage_dir"]):
            os.makedirs(configs["storage_dir"])

        if args.train:
            if os.path.isdir(os.path.join(configs["storage_dir"], args.model_name)) and not args.overwrite:
                raise FileExistsError("There's an existing model with the same name. Specify --overwrite. Aborted.")
            train_model(configs, args.model_name)

        if args.eval:
            eval_model(configs, args.model_name)


if __name__ == "__main__":
    main()
