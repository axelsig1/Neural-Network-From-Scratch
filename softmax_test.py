"""
Softmax activation implementation (module-level docstring)
This module implements the softmax activation for a batch of network outputs (logits).
Softmax converts raw scores for each class into probabilities that sum to 1 for each
sample. The typical steps are:
- shift the logits for numerical stability,
- exponentiate the shifted logits,
- sum the exponentials per sample,
- divide each exponential by its sample's sum to obtain normalized probabilities.
Why we subtract the largest value (long explanation)
----------------------------------------------------
Directly computing np.exp(x) for arbitrary logits x can produce numerical problems:
- For large positive values of x, exp(x) can overflow to +inf in finite-precision
    arithmetic (float32/float64), making probabilities NaN or unusable.
- For very negative values, exp(x) can underflow to 0, which is less harmful but
    still reduces numerical accuracy when mixed with much larger exponentials.
To eliminate the overflow risk while preserving the correct probabilities, we
exploit that softmax is invariant under adding (or subtracting) the same constant
to all elements of a given sample's logits. Formally, for a vector x and any
constant c:
        softmax_i(x) = exp(x_i) / sum_j exp(x_j)
                                 = exp(x_i + c) / sum_j exp(x_j + c)
                                 = exp(x_i - m) / sum_j exp(x_j - m)
where choosing c = -m and m = max_j x_j makes the largest shifted logit equal
to zero and all other shifted logits <= 0. This choice has two important effects:
1. The largest value becomes 0, so exp(0) = 1, and all other exponentials are
     in the interval (0, 1], avoiding extremely large exponentials that could
     overflow.
2. Because we subtract the same scalar from every element of a sample, the
     relative ratios between exponentials are unchanged, and normalization yields
     identical probabilities to the unshifted computation (up to floating-point
     rounding). Multiply numerator and denominator by exp(-m) demonstrates this:
        exp(x_i)/sum_j exp(x_j) = [exp(x_i) * exp(-m)] / [sum_j exp(x_j) * exp(-m)]
                                                     = exp(x_i - m) / sum_j exp(x_j - m)
Practical details
-----------------
- Use the maximum per sample (axis=1 for a 2D array shaped (n_samples, n_classes))
    to shift logits independently for each sample in the batch.
- Use keepdims=True when computing the max so the result keeps the same number
    of dimensions and can be broadcast-subtracted from the logits without shape errors.
- After exponentiation, sum across the same axis (axis=1) with keepdims=True to
    get the per-sample denominators for correct broadcasting during normalization.
- Verify that the resulting probabilities sum to 1 per sample (up to tiny
    floating-point error). This is a useful sanity check during development.
- Prefer higher precision (float64) when debugging numerical issues; in production
    deep learning float32 is common and the max-shift trick is essential there.
In short: subtracting the per-sample maximum is a cheap and mathematically safe
transformation that prevents overflow, preserves the correct softmax output, and
improves numerical stability and robustness of training and inference.
"""


import numpy as np
import nnfs

# Initialize nnfs library (sets random seed and configures global settings for reproducibility)
nnfs.init()

layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

exp_values = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True))


#print(np.sum(layer_outputs, axis=1, keepdims=True)) # summing each row, keepdims to maintain the 2D structure


norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
print(np.sum(norm_values, axis=1))