cd `dirname $0`

wget "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
python preprocessing_sample_iris.py

echo "finished!!"
echo "Please execute training command,\n for example: python classifier.py -f dataset.csv -A 4 --feature_selection --grid_search --output_json test.json --output_csv test.csv"
