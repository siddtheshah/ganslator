# GANSlator

This project features a generative adversarial network used to generate modified speech waveforms.
Originally intended for language translation problems, it is currently being used on the simpler problem of
emotion transformation.

## Dependencies
`pip3 install -r requirements.txt`

## Dataset Download And Preprocessing
`sudo apt install sox`
`sh get_ravdess.sh`
`sh process_ravdess.sh`

## Train and Eval model

1. Modify config.json as desired
2. `python3 main.py --model_name=my_model --train --eval --dataset ravdess_chunked`

## Results

Awaiting.