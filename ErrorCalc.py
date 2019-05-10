import statistics as stat
import json
import numpy as np
from math import atan2, degrees

test_name = 'test1'
file_name = 'tests/' + test_name + '_accuracy.json'
with open(file_name) as f:
    data = json.load(f)

correct_translation = np.array((0.0, 0.0))
# rotation in degree
correct_rotation = -90

models = data['models']
n = len(models)

estimated_translations = []

mae_trans = 0.0
for model in models:
    estimated_translation = np.array((model[0][2], model[1][2]))
    estimated_translations.append(estimated_translation)
    mae_trans += np.linalg.norm(correct_translation - estimated_translation)
mae_trans = mae_trans / n

avg_detected_trans = np.mean(estimated_translations, axis=0)
sd_detected_trans = np.std(estimated_translations, axis=0)


mae_rot = 0.0
estimated_rotations = []
for model in models:
    A11 =  model[0][0]
    A21 = model[1][0]

    if A11 > 1:
        A11 = 1
    if A11 < -1:
        A11 = -1

    if A21 > 1:
        A21 = 1
    if A21 < -1:
        A21 = -1
    angle = degrees(atan2(A21,A11))
    estimated_rotations.append(angle)
    mae_rot += np.absolute(correct_rotation - angle)
mae_rot = mae_rot / n

avg_detected_rot = np.mean(estimated_rotations, axis=0)
sd_detected_rot = np.std(estimated_rotations, axis=0)


inlier_rates = data['inlier_rates']
avg_inlier_rate = stat.mean(inlier_rates)
sd_inlier_rate =  stat.stdev(inlier_rates)

stats = {}
stats['number_keypoints_reference'] = data['number_keypoints_reference']
stats['number_keypoints_comparison'] = data['number_keypoints_comparison']
stats['number_matches'] = data['number_matches']
stats['match_rate'] = data['match_rate']
stats['avg_detected_translation'] = avg_detected_trans.tolist()
stats['sd_detected_translation'] = sd_detected_trans.tolist()
stats['avg_detected_rotations'] = avg_detected_rot
stats['sd_detected_rotations'] = sd_detected_rot
stats['mean absolute error translation'] = mae_trans
stats['mean absolute error rotation'] = mae_rot
stats['avg_inlier_rate'] = avg_inlier_rate
stats['sd_inlier_rate'] = sd_inlier_rate


test_name = 'test1'
file_name = 'tests/' + test_name + '_stats.json'

with open(file_name, 'w') as outfile:
    json.dump(stats, outfile)
