"""Categorical Cross-Entropy Loss Calculation
Classes:      3
Label:        0
One-hot:      [1, 0, 0]
Prediction:   [0.7, 0.1, 0.2]
L = -Σ(y_i * log(p_i)) = -(1*log(0.7) + 0*log(0.1) + 0*log(0.2)) = -log(0.7) = 0.3567


softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

calsses = [0, 1, 2]  dog, cat, human

class_targets = [dog, cat, cat]  -> [0, 1, 1]

correct_confidences = [0.7, 0.5, 0.9]

"""

import numpy as np

softmax_outputs = [0.7, 0.1, 0.2]
target_outputs = [1, 0, 0]

loss = -np.sum(target_outputs * np.log(softmax_outputs))
print(loss)

print('---')

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]  # dog, cat, cat

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(neg_log)
average_loss = np.mean(neg_log)
print(average_loss)