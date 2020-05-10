mkdir -p download/ravdess
mkdir -p data/ravdess
DOWNLOAD_URL="https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
wget $DOWNLOAD_URL download/ravdess

CONVERTED_PATH=data/ravdess

for filename in download/ravdess/**/*; do
  file_base="$(basename -- $filename)"
  echo $file_base
	sox --ignore-length $filename -r 16000 "${CONVERTED_PATH}${file_base}"
done
