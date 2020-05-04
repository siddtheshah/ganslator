# GANSlator

This project features a generative adversarial network used to generate modified speech waveforms.
Originally intended for language translation problems, it is currently being used on the simpler problem of
emotion transformation.

## Dependencies
1. Tensorflow 2.0
2. Python 3.5

## Dataset Download

1. Install kaggle API
2. `kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data/ravdess`

## Train and Eval model

1. Modify config.json as desired
2. `python3 main.py --model_name=my_model --train --eval`

## Results
