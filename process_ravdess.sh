for filename in data/ravdess/**/*; do
	echo $filename
	sox --ignore-length $filename -r 16000 $filename
done
