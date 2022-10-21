# SoftmaxRegression example
simple functions for performing classification of n > 2 classes

## Installation

```julia
# get into the package manager by pressing the ] key on keyboard
add https://github.com/YSanchezAraujo/SoftmaxRegression.jl
```


## Example usage
```julia
using SoftmaxRegression;

# assuming you have X (design matrix) and y  (class label vector)

est = fit_softmax(X, y);

# a version with an l2-penalty
est = fit_softmax(X, y, 0.7);

# est will be a tuple, check it's names to get accuracy, predictions, estimated coefficient weights (betas), ect. 
```
