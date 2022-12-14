"""
inputs (required): 
    params: vector (contains n_intercepts + length(vec(betas)) parameters)
            this is a concatenated 1-d vector of all parameters you wish the estimate
            the intercept terms should be first, and you should have n_class of them. 
            After comes the vectorized coefficient matrix\n
    X: your design matrix, global intercept is assumed to be included (column of ones)\n
    y: target class labels\n
    lam: l2 (ridge) penalty. Larger values of lam will result in more regularization
outputs: 
    negative log-likelihood which is optionally penalized
"""
function softmax_regression_negloglik(params, X, y, lam)
    n_samp, n_cols = size(X)

    n_class = length(unique(y))

    intercepts = params[1:n_class]

    betas = reshape(params[n_class+1:end], (n_cols, n_class))

    vals = intercepts' .+ X * betas

    loglik = 0

    for k in 1:n_samp
        
        p = softmax(vals[k, :])
        
        loglik = loglik + logpdf(Categorical(p), y[k])

    end

    w = vec(betas)

    pen = lam / (2*n_samp) * w'w

    return -loglik + pen
end

"""
inputs (required): 
    X: your design matrix, global intercept is assumed to be included (column of ones)\n
    y: target class labels\n
inputs (optional): 
    lam: l2 (ridge) penalty. Larger values of lam will result in more regularization\n
    verbose: bool deciding if the success or failure of optimization will be printed out\n
outputs: 
   tuple with estimated intercepts and coefficient matrix: 
     res = softmax_regression(X, y)
     res.intercepts # to access the estimated intercepts
     res.betas # to access the coefficient matrix
"""
function softmax_regression_opt(X, y, lam, verbose=true)
    n_samp, n_cols = size(X)

    n_class = length(unique(y))

    # initialize parameters
    betas = rand(n_cols, n_class)

    intercepts = rand(n_class)

    beta_arr = vec(betas)

    all_params = [intercepts; beta_arr]

    opt = optimize(w -> softmax_regression_negloglik(w, X, y, lam), all_params, BFGS(); autodiff=:forward)

    if verbose
        println(opt)
    end

    b0s = opt.minimizer[1:n_class]

    beta_est = reshape(opt.minimizer[n_class+1:end], (n_cols, n_class))

    return (intercepts = b0s, betas = beta_est)
end
        
"""
 inputs (required): 
    X: design matrix for test or train set samples\n
    y: array of class labels\n
    probs: predicted class probabilties
outputs: 
   tuple with variance-covariance matrix for coefficient estimates
   associated with each class and the standard error estimates associated with each class\n
"""
function var_estimates(X, y, probs, lam)
    n_class = maximum(y)
                        
    W_per_class = [
        diagm(probs[:, j] .* (1 .- probs[:, j]))
        for j in 1:n_class
    ]
                
    V = [pinv(X' * W_per_class[j] * X) for j in 1:n_class]
    
    stderrs = hcat([sqrt.(diag(V[j])) for j in 1:n_class]...)
    
    return (vcov = V, stderr = stderrs)
            
end
