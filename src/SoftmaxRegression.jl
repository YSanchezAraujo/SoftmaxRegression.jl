module SoftmaxRegression

include("fit.jl")
        
export 
        softmax_regression_negloglik, softmax_regression_opt, predict_softmax_opt,
        var_estimates, fit_softmax
        

end # module
