# SoftmaxRegression example
simple functions for performing classification of n > 2 classes

## Installation

```julia
# get into the package manager by pressing the ] key on keyboard
add https://github.com/YSanchezAraujo/SoftmaxRegression.jl
```

```julia
# assuming you have X (design matrix with global intercept) and y  (class label vector)

ests = softmax_regression_opt(X, y; lam=0.7, verbose=true);

preds = predict_softmax_opt(X, ests.intercepts, ests.betas);

# assess classification accuracy
mean(preds.c .== y)

# if you want to look at class probabilities check preds.p

```
