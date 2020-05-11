CONVERTED_PATH=data/ravdess/
mkdir -p data/ravdess
for filename in download/**/*; do
  file_base="$(basename -- $filename)"
  echo $file_base
	sox --ignore-length $filename -r 16000 "${CONVERTED_PATH}${file_base}"
done
