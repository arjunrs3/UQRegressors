# uqregressors.conformal.k_fold_cqr

This class implements conformal quantile regression in a K-fold manner to obtain uncertainty estimates which are often conservative, but use the entire dataset available. 
This can result in large improvements over split conformal quantile regression, particularly in cases where the dataset is sparse. 

!!! tip
    The quantiles of the underlying quantile regressor can be tuned with the parameters tau_lo and tau_hi as in the paper. This can often result in more efficient intervals. 

!!! note
    K-fold Conformal Quantile Regression can be overly conservative in prediction intervals, particularly in sparse data settings or when the underlying estimator has high variance.

::: uqregressors.conformal.k_fold_cqr