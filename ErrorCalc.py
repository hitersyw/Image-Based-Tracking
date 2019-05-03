import statistics as stat
# for model comparison
from sklearn.metrics import mean_absolute_error
import json

test_name = 'test1'
file_name = 'tests/' + test_name + '_accuracy.json'
with open(file_name) as f:
    data = json.load(f)

number_keypoints_reference = data['number_keypoints_reference']
number_keypoints_comparison = data['number_keypoints_comparison']
number_matches = data['number_matches']
match_rates = data['match_rates']
models = data['models']
inlier_rates = data['inlier_rates']

input_list =
mean = stat.mean(input_list)
sd =  stat.stdev(input_list)
