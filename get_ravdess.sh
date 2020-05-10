mkdir -p download
mkdir -p data/ravdess
DOWNLOAD_URL="https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
cd download
wget $DOWNLOAD_URL
unzip "Audio_Speech_Actors_01-24.zip?download=1"
