import tensorflow as tf
import numpy as np
import network.combined_model as cm

import os
import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train', default=False, action='store_true', help='Train a new model')
parser.add_argument('--eval', default=False, action='store_true', help='Run eval on the model')
parser.add_argument('--model_name', help='Specify a run name. (Required)', action='store')
args = parser.parse_args()

def train_model(configs, model_name):
    model = cm.GANslator(r_scale=configs['r_scale'], feature_size=configs['mel_bins'], filter_dim=configs['filter_dim'])

    print("Model finished training.")


def eval_model(configs, model_name):

    model = tf.keras.models.load_model(os.path.join(configs["storage_dir"], model_name))


def main():
    with open('config.json') as config_file:
        configs = json.load(config_file)

        if args.train:
            train_model(configs, args.model_name)

        if args.eval:
            eval_model(configs, args.model_name)

if __name__ == "__main__":
    main()