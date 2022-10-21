using StatsBase: mean;
using Distributions: logpdf, Categorical;
using Optim;
using NNlib: softmax;
using LinearAlgebra: pinv, diagm, diag;

include("optim_base_class_fns.jl");
include("optim_penalty_fns.jl");

# WARNING, UNTESTED

"""
inputs (required): 
    X: design matrix for test or train set samples\n
    intercepts: estimated intercepts as a 1-d array\n
    betas: estimated coefficent matrix
outputs: 
   tuple with class labels and class probabilities
"""
function predict_softmax_opt(X::Matrix, intercepts, betas)

    vals = [zeros(size(X, 1)) intercepts' .+ X * betas]

    probs = mapslices(softmax, vals, dims=2)

    class_index = mapslices(argmax, vals, dims=2)

    return (c = vec(class_index), p = probs)
end

"""
inputs (required): 
    X: design matrix for test or train set samples\n
    intercepts: estimated intercepts as a 1-d array\n
    betas: estimated coefficent matrix
outputs: 
   tuple with class labels and class probabilities
"""
function predict_softmax_opt(X::Matrix, intercepts, betas, lam=nothing)

    vals = intercepts' .+ X * betas

    probs = mapslices(softmax, vals, dims=2)

    class_index = mapslices(argmax, vals, dims=2)

    return (c = vec(class_index), p = probs)
end

"""
inputs (required)    
    X: your design matrix, global intercept is assumed to be included (column of ones)\n
    y: target class labels\n
inputs (optinal)
    lam: float, penalty parameter. Larger values of lam result in more penalization
    verbose: bool, whether or not to show the result of optimization
"""
function fit_softmax(X, y, lam = nothing, verbose = true)
  
  if isnothing(lam) || lam == 0
      println("using y == 1 as reference class")
    
      ests = softmax_regression_opt(X, y; verbose = verbose)

      preds = predict_softmax_opt(X, ests.intercepts, ests.betas)

      vcv = var_estimates(X, y, preds.p)
        
  else
      ests = softmax_regression_opt(X, y, lam; verbose = verbose) 
      
      preds = predict_softmax_opt(X, ests.intercepts, ests.betas; lam=lam)

      vcv = var_estimates(X, y, preds.p, lam)

  end
          

  acc = mean(preds.c .== y)
        
   return (
          betas = ests.betas,
          intercepts = ests.intercepts,
          acc = acc,
          yhat = preds.c,
          probs = preds.p,
          vcov = vcv.vcov,
          stderr = vcv.stderr
          )
  
end
