# uqregressors.conformal.conformal_ens

This method employs normalized conformal prediction as described in [Tibshirani, 2023](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf). 
The difficulty measure, \(\sigma\), used for normalization is taken to be the standard deviation of the predictions of all models in an ensemble, while the ensemble mean 
is returned as the mean prediction. 

::: uqregressors.conformal.conformal_ens

