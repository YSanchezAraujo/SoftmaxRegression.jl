using Distributions: logpdf, Categorical;
using Optim;
using NNlib: softmax;
using LinearAlgebra: pinv;



"""
inputs (required): 
    params: vector (contains n_intercepts + length(vec(betas)) parameters)
            this is a concatenated 1-d vector of all parameters you wish the estimate
            the intercept terms should be first, and you should have n_class - 1 of them. 
            After comes the vectorized coefficient matrix\n
    X: your design matrix, global intercept is assumed to be included (column of ones)\n
    y: target class labels\n
outputs: 
    negative log-likelihood which is optionally penalized
"""
function softmax_regression_negloglik(params, X, y)
    n_samp, n_cols = size(X)

    n_class = length(unique(y))

    intercepts = params[1:n_class-1]

    betas = reshape(params[n_class:end], (n_cols, n_class-1))

    vals = intercepts' .+ X * betas

    loglik = 0

    for k in 1:n_samp
        
        p = softmax([0; vals[k, :]])
        
        loglik = loglik + logpdf(Categorical(p), y[k])

    end

    return -loglik
end

"""
inputs (required): 
    X: your design matrix, global intercept is assumed to be included (column of ones)\n
    y: target class labels\n
inputs (optional): 
    verbose: bool deciding if the success or failure of optimization will be printed out\n
outputs: 
   tuple with estimated intercepts and coefficient matrix: 
     res = softmax_regression(X, y)
     res.intercepts # to access the estimated intercepts
     res.betas # to access the coefficient matrix
"""
function softmax_regression_opt(X, y; verbose=true)
    n_samp, n_cols = size(X)

    n_class = length(unique(y))

    # initialize parameters
    betas = rand(n_cols, n_class - 1)

    intercepts = rand(n_class - 1)

    beta_arr = vec(betas)

    all_params = [intercepts; beta_arr]

    opt = optimize(w -> softmax_regression_negloglik(w, X, y), all_params, BFGS(); autodiff=:forward)

    if verbose
        println(opt)
    end

    b0s = opt.minimizer[1:n_class-1]

    beta_est = reshape(opt.minimizer[n_class:end], (n_cols, n_class-1))

    return (intercepts = b0s, betas = beta_est)
end
