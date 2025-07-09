# Uncertainty Estimation Methods

## Uncertainty Estimation with Distributional Methods

For generating the uncertainty intervals returned by a function, we make a distinction between distributional methods and non-distributional methods. Distributional methods seek to return a distribution instead of a point prediction for each point, whereas non-distributional methods return an upper and lower bound with some confidence \(1-\alpha\). Many distributional methods exist, but the below methods were selected for their published performance on non-aerospace datasets.

### Standard Gaussian Process Regression

Gaussian process regression is perhaps one of the most established and widely used methods for regression with uncertainty quantification within the aerospace discipline. A brief introduction is given here to standard Gaussian process regression, with a full treatment in [Rasmussen]. Gaussian processes model a distribution over candidate functions which fit the training data by assuming that the training outputs and the test outputs are drawn from a jointly distributed Multi-variate Gaussian. Neglecting measurement noise and assuming a zero mean prior for simplicity of explanation, we assume that the joint distribution of the training outputs, \(y_{tr}\) and the testing outputs, \(y_{te}\), is given by:

\[
\begin{bmatrix}
    f(X_{tr}) \\
    f(X_{te})
\end{bmatrix} = \mathcal{N}\left(0,
\begin{bmatrix}
    K &  K_* \\
    K_*^{T} & K_{**}
\end{bmatrix} \right)
\]

for covariance matrices given by some measure of similarity (kernel function: \(\text{cov}\)) \(K=\text{cov}(X_{tr}, X_{tr}), K_* = \text{cov}(X_{tr}, X_{te}), K_{**} = \text{cov}(X_{te}, X_{te})\). After obtaining the data, \(f(X_{tr}) = y_{tr}\), we obtain the posterior distribution by conditioning this multivariate normal distribution:

\[
y_{te} | y_{tr}, X_{tr}, X_{te} = \mathcal{N}\left(K_*K^{-1}y_{tr}, K_{**} - K_*K^{-1}K_*^T\right)
\]

As an intuitive explanation of this formula, the mean of this normal distribution is the outputs of the training data weighted by how similar their corresponding inputs are to the test input, while the variance is entirely dependent on this similarity. Similarity is determined by the kernel function mentioned above, which uses the kernel trick to evaluate the similarity of the input vectors in a much higher dimensional space to evaluate a rich measure of similarity. One of the most commonly used kernel functions is the Radial Basis Function (RBF), which is computed for the length scale parameter, \(l\) as:

\[
\text{cov}(x_i, x_j) = \exp\left(-\frac{||x_i-x_j||^2_2}{2l^2}\right).
\]

This kernel function is used for several desirable properties; namely its infinite dimensional feature space, exponentially decaying value of similarity, and positive definiteness. One limitation of this kernel function is that it implicitly assumes that the underlying function is smooth, and can give uninformative measures of similarity in high dimension, where the Euclidean distance between points tends to concentrate.

This kernel is also strongly dependent on a reasonable choice of the length-scale parameter \(l\), which is often optimized by maximizing the log-likelihood of the training data, \(y\) with gradient based methods. The log-likelihood is written as:

\[
\log p(y_{tr} | X_{tr}, l) = -\frac{1}{2}y_{tr}^T K^{-1}y_{tr} - \frac{1}{2} \log|K| - C_1
\]

The first term of this measures the relation between the output values, and the relation expected by the covariance matrix structure. For example, if the covariance matrix predicts that y values should be highly correlated and they are not, the loss will be large. The second term is a measurement complexity term, which tries to drive the outputs of the kernel covariance lower. In the case of the RBF kernel, this corresponds to prioritizing smooth functions which still have a good data-fit. The constant, \(C_1\) is not important for the hyperparameter optimization. In practice, a measurement noise term, \(\sigma_n^2\) is co-optimized with \(l\), but it is not included here for simplicity of explanation.

Gaussian process regression with hyperparameter optimization can often fit low-dimensional smooth functions extremely well, but also has a time complexity of \(O(n^3)\) to fit data arising from the need to invert the \(n \times n\) covariance matrix. This makes Gaussian process regression a natural choice for datasets with only a few number of points. After being trained, the time complexity to perform inference on \(m\) points is \(O(n^2m)\), meaning that Gaussian process regression can quickly become impractical for optimization routines or other use cases where the number of predictions required is extremely large. Typically, standard Gaussian process regression is impractical for datasets beyond a few thousand points.

### Black-Box Matrix Multiplication Gaussian Process Regression

The key reason why Gaussian Process Regressions have a large training time complexity is the requirement to invert the matrix \(K\). This is typically done using the Cholesky decomposition, but there exist other methods of approximate inversion which can significantly reduce the training time complexity. In particular, the black-box matrix multiplication variation of a GP uses \(T\) iterations of the conjugate gradients algorithm with a low rank preconditioner in order to compute an approximate inversion in \(O(n^2T)\) time, giving an exact solution if \(T=n\) [Gardner]. Additionally, they have optimized their methodology for parallel computing hardware, with packaged code available in the python repository GPytorch. Since all of the deep learning models to follow are implemented for GPU hardware using PyTorch, this allows for a fair comparison of training and inference time.

### Monte-Carlo Dropout

Monte-Carlo dropout is a method of uncertainty quantification for arbitrary neural networks, which is a neural network approximation of Gaussian processes. Critically, because of the deep learning architecture, it can avoid the large time complexity of Gaussian Processes such that it can scale to extremely large datasets. To train a neural network with dropout, at each iteration, each neuron in the hidden layers is assigned a binary variable with probability \(p_{drop}\) (typically near 0.1 or 0.2) of being 0. Each neuron assigned a value of 0 is dropped for that training pass, meaning that it simply returns a value of zero. The parameters are trained to minimize the mean squared error loss in this stochastic setting, where each iteration has a different combination of parameters. To perform inference, we perform \(T\) passes through the network with the same dropout methodology as during training to obtain \(T\) different samples of function values at each point. If desired, a distribution (typically Gaussian) can be fit to the outputs at each point. The theoretical results of [Gal] demonstrate that this approximates Bayesian Neural networks, where a distribution is placed over the weights within the neural network. However, Bayesian neural networks require a doubling of the parameters of a standard neural network, or one with dropout, and can also have undesirable optimization landscapes with poor convergence properties. The empirical results of [Gal] demonstrate comparable or better performance of Monte-Carlo dropout as compared to Bayesian Neural Networks on a variety of datasets, which is why Monte-Carlo dropout is implemented over Bayesian Neural Networks for this comparison.

While Monte-Carlo dropout is an extremely efficient method of obtaining uncertainty distributions in addition to a mean prediction, the outputs are random, so they can be non-smooth, which can cause difficulties for optimization. Additionally, each training iteration optimizes different network parameters for a new random realization of the mean function, so the accuracy of this method may be less than that of traditional neural networks without dropout. Key benefits of Monte-Carlo dropout include that there are no required assumptions on the distribution of uncertainty, and that implementation requires minimum modification to standard neural networks.

### Deep Ensemble

Deep Ensembles, proposed by [Lakshminarayanan], are another relatively inexpensive approximation of Bayesian Neural Networks, which involve training several (i.e., an ensemble) of models and combining their outputs. Each regressor within the ensemble assumes that each point is an observation from a Gaussian distribution, and is trained to predict both the mean and standard deviation of an output given an input. During training, the negative log likelihood of the data given the predicted mean and standard deviation is minimized. During inference, the mean and variance from each regressor are combined by assuming that the overall distribution of uncertainty is also Gaussian with the mean and variance of the mixture. For \(K\) models, the mean and standard deviation are calculated as:

\[
\mu_*(x) = K^{-1}\sum_{k=1}^K \mu_{\theta_k}(x), \sigma_*^2(x)=K^{-1}\sum_{k=1}^K(\sigma_{\theta_m}^2(x) + \mu_{\theta_m}^2(x)) - \mu_*^2(x)
\]

When implementing this model, it is useful to let the outputs of each model predict the log of the standard deviation instead of the standard deviation itself for numerical stability and positive definiteness of the standard deviation. Additionally, one variant of this model is to randomly select a different subset of the training examples for every regressor to prioritize diversity in prediction and calibrate the uncertainty intervals towards difficulty of prediction. However, this was not necessary in order to produce well-calibrated intervals during the empirical studies by [Lakshminarayanan]. In these studies, Deep Ensembles were shown to perform similarly or slightly worse as compared to Monte-Carlo dropout on a variety of public regression datasets on the basis of RMSE, but to significantly outperform Monte-Carlo dropout on the basis of negative log likelihood evaluated on the test set. This implies that deep ensembles output well calibrated uncertainty distributions, as the log-likelihood will penalize both poor fit and either over or under confident uncertainty intervals. This is explained in the paper by the fact that deep ensembles explicitly optimize for negative log likelihood during training as opposed to mean squared error.

Deep ensembles are able to produce smooth uncertainty estimates in a well-calibrated sense, but assume that the uncertainty follows a known distribution. In many cases, a Gaussian distribution may be a good approximation to the distribution of uncertainty, unless prior knowledge about the problem indicates that a different distribution should be used. Additionally, Deep ensembles require fitting \(K\) estimators, although they can be fit in parallel to accomplish the same running time as standard neural networks. Ensemble predictors are generally desirable, as they can reduce the variance of the output function found through training, particularly for sparse datasets in high dimensions. More detail on the variance reduction of ensemble predictors is described in Variance reduction with ensemble methods section below.

## Uncertainty Estimation with Distribution-free Methods

While assuming a Gaussian distribution on uncertainty is often a reasonable assumption, this may not hold for certain processes or certain datasets. As a simple example, if we consider the drag of an airfoil near stall, very small changes in the angle of attack can result in large drops in lift. Therefore, slight modeling errors will result in a non-Gaussian distribution of error, where errors are more likely to be distributed below the prediction, to have a long-tailed distribution, and perhaps even demonstrate bi-modality. This can become especially important if the observations come from the physical world, as in a wind tunnel test, which may have measurement noise which does not demonstrate a Gaussian distribution. For this reason, a method of quantifying uncertainty which does not assume a distribution on uncertainty is included in these comparisons. A method which is rising in popularity in the machine learning community to accomplish distribution-free uncertainty estimation is conformal prediction. An overview of the simplest method of conformal prediction, split conformal prediction, followed by a description of more sophisticated methods.

The key idea of conformal prediction is to return a set, \(\hat C (X_{te_i})\), for each test input, such that:

\[
\mathbb{P}(y_{te_i} \in \hat C(X_{te_i})) \geq 1- \alpha,
\]

where \(\alpha\) is a user-specified error rate. In this way, the model developer has a \(1-\alpha\) level of confidence that the returned intervals contain the true response values [Vovk2022]. While several methods of conformal prediction exist, we explain the process with the simplest implementation: split conformal prediction.

### Split Conformal Prediction

Conformal prediction returns statistically valid sets with minimal assumptions on the data and no assumptions on the predictor by leveraging ideas from rank statistics. First, the available data is partitioned into a training, \(S_1\), and calibration, \(S_2\), set. A point predictor, \(\hat{f}\), is fit on the training data, and absolute residuals are found on the calibration set. The absolute residuals are of the form:

\[
R_i=|y_i-\hat{f}(X_i)|, \ i \in S_2.
\]

For a feature-response pair in the test set with residual \(R_{te_i}\), we expect that the rank (or the sorted position) of the calibration residuals and the test residual is evenly distributed. This assumes that the new residual is exchangeable with the calibration residuals. We can then form the rank-adjusted quantile:

\[
\hat{q} = \text{the } \frac{\lceil(n_2+1)(1-\alpha)\rceil}{n_2} \text{ quantile of } R_i, \ i \in S_2,
\]

where the term \(\frac{(n_2 + 1)}{n_2}\) is a finite sample correction such that:

\[
\mathbb{P}(R_{te_i} \leq \hat{q})  \geq 1-\alpha.
\]

A prediction interval satisfying the above can then be formed as:

\[
\hat C(X_{te_i}) = [\hat{f}(X_{te_i}) - \hat{q},\hat{f}(X_{te_i})+\hat{q}],
\]

The only assumption made about the data and the predictor is that the residuals of the calibration set and the residuals of new test points are exchangeable, meaning that their joint probability distribution is the same regardless of their permutation. Split conformal prediction is a simple method of generating statistically valid prediction intervals, but it requires partitioning the available data into a training and calibration set, which means that not all of the available data is used to train the model (low statistical efficiency) [Angelopoulos2023].

The size of the calibration set also has a substantial impact on the performance of split conformal prediction. Although the average coverage is guaranteed to be \(1-\alpha\), the total coverage over different random splits of the data follows a beta distribution with approximate variance \(\frac{\alpha(1-\alpha)}{n_2+2}\). Therefore, if \(n_2\) is small, a given split of the available data can substantially undercover. In practical implementations of split conformal prediction, it is recommended to use calibration sets of at least 100 feature response pairs. To remedy the low statistical efficiency of this method, ensemble regression can be used through the methodology of K-fold CV+.

### K-fold CV+ and CV-minmax

The exchangeability requirement mandates determining the residuals on a set of data not used to train the model. K-fold CV is a method of conformal prediction based on ideas from cross-validation, where the data is split into \(K\) folds, and \(K\) predictors are trained on each permutation of \(K-1\) folds. For each predictor, calibration residuals are evaluated on the fold that is left out from training. In this way, a set of calibration residuals which are exchangeable with the residuals from unseen data can be generated while still using a majority of data for model training. The interval half-width can then be constructed as in split conformal prediction, using the combined set of calibration residuals from each fold [barber2020].

Two methods of generating prediction intervals with K-fold CV provide statistical guarantees. CV+ centers the prediction interval around the average of the \(K\) predictors:

\[
\hat C(X_{n+1}) = [\hat{f}_{mean}(X_{n+1}) - \hat{q},\hat{f}_{mean} (X_{n+1})+\hat{q}],  \ \hat{f}_{mean} (X_{n+1}) = \frac{1}{K} \sum_{i=1}^K \hat{f}_i(X_{n+1}),
\]

whereas CV-minmax constructs bounds around the minimum and maximum of the \(K\) predictors:

\[
\hat C(X_{n+1}) = [\hat{f}_{low}(X_{n+1}) - \hat{q},\hat{f}_{high} (X_{n+1})+\hat{q}],  \ \hat{f}_{low} (X_{n+1}) = \text{min}(\hat{f}_i), \ \hat{f}_{high} (X_{n+1}) = \text{max}(\hat{f}_i), \ i \in K.
\]

In exchange for higher statistical efficiency, CV+ has a relaxed statistical guarantee. The average coverage guarantee of CV+ is \(1-2\alpha - \sqrt{\frac{2}{n_2}}\), while the average coverage guarantee of CV-minmax is \(1-\alpha\). In practice, however, CV+ generally has an average coverage of \(1-\alpha\), while the coverage of CV-minmax is typically larger than \(1-\alpha\).

Thus, while CV+ does not enjoy the same guarantee as split conformal prediction, it typically performs well in practice [barber2020]. CV-minmax always returns prediction intervals which are wider than CV+, and tends to overcover in practice.

An additional downside of generic split conformal prediction is that the interval half-width, \(\hat{q}\), is constant across the output space, meaning that the intervals will generally be too small for input values corresponding to outputs which are difficult to predict, and too large for easily predictable outputs, despite maintaining a \(1-\alpha\) average coverage [Romano2019]. A constant width is uninformative when it comes to adaptive sampling, so different methods of conformal prediction modify the methodology to generate adaptive interval widths.

### Conformal Quantile Regression

Conformal quantile regression produces locally adaptive prediction intervals by wrapping conformal prediction around an interval predictor instead of a point predictor. Instead of fitting a predictor to the mean of the available data, two predictors are fit to the \(\alpha/2\) and the \(1-\alpha/2\) quantiles of the data. This alone does not provide any statistical guarantees, but the split or CV conformal prediction methods can then be applied to the quantile estimators to produce modifications to the interval width that satisfy the desired statistical guarantee [Romano2019].

For many regression methods, fitting predictors to quantiles of the data can be done by modifying the loss function. In this work, neural networks are used with a tilted loss function to predict quantiles. The neural network predicts a quantile, \(\tau\) by minimizing the loss, \(L\) over all points, where:

\[
L(X_i)=\max\left(-\tau (\hat{f}(X_i)-Y_i), (1-\tau)(\hat{f}(X_i)-Y_i)\right).
\]

A simple synthetic dataset to demonstrate the use of wrapping a quantile regressor with conformal prediction was created. A set of 1,000 data pairs was generated with:

\[
y=0.1(\sin(2\pi x) + 0.4 \epsilon (0.1 + x)), \ x \in[0,1], \ \epsilon \sim \mathcal{N}(0, 1).
\]

Defining \(\alpha=0.1\), neural networks were fitted to the 0.05 and 0.95 quantiles using 250 training points. Split conformal prediction was used with a calibration set size of 250. Finally, the coverage was evaluated on a set of 500 test points. The prediction intervals and test points are shown in the figure below. The conformal prediction interval bounds are slightly larger than the prediction made by the quantile neural network, which ensures the statistical guarantee is met. The coverage was evaluated over 10,000 random splits of the data, shown in the histogram below. The histogram demonstrates that the average coverage over these trials meets the desired coverage of 0.9. The variance of the coverage could be reduced by increasing the number of calibration points used.

![Prediction interval visualization](sin_function.png)
![Coverage evaluated over 10000 trials](sin_function_coverage.png)

By wrapping quantile regressors with CV+ or CV-minmax, statistically-valid and locally-adaptive prediction intervals can be generated with relatively high statistical efficiency compared to split conformal prediction. The results presented in this work use quantile regressors wrapped with CV+. K-fold CV+ enables locally adaptive interval widths with relatively high statistical efficiency, but will still struggle to produce well-calibrated with datasets with less than a couple hundred points, as the small calibration set size will result in a high-variance distribution of coverage and interval widths. Additionally, since optimization of the mean function does not occur directly during training, the RMSE of this method as compared to other methods may be higher.

### Normalized Conformal Prediction (Conformal Ensemble)

Using quantile regression is an effective method of obtaining locally adaptive interval widths when there are enough training points to give the model information about quantiles of the dataset. Another method of conformal prediction is known as normalized conformal prediction, which follows the same methodology as split conformal prediction, but constructs adaptive intervals based on normalized non-conformity scores. The key idea is that instead of choosing \(\hat{q} = \text{the } \frac{\lceil(n_2+1)(1-\alpha)\rceil}{n_2} \text{ quantile of } R_i, \ i \in S_2\), we choose

\[
\hat{q} = \text{the } \frac{\lceil(n_2+1)(1-\alpha)\rceil}{n_2} \text{ quantile of } \frac{R_i}{\sigma(X_i)}, \ i \in S_2,
\]

where \(\sigma(X_i)\) is an estimator of the difficulty of predicting \(y_i\). For points which are difficult to predict, \(\sigma(X_i)\) should be large and vice versa, such that the normalized residuals, \(\frac{R_i}{\sigma(X_i)}\), are approximately equal. To construct the interval for a point in the test interval, we first compute \(\sigma(X_{te_i})\), and then construct the interval as:

\[
\hat C(X_{te_i}) = [\hat{f}(X_{te_i}) - \hat{q}\sigma(X_{te_i}),\hat{f}(X_{te_i})+\hat{q}\sigma(X_{te_i})]
\]

In this way, if \(\sigma\) is a well calibrated estimate of prediction difficulty, conformal prediction will return adaptive interval widths. While this method trains a mean regressor and can often be easier to implement than conformal quantile regression, it loses statistical guarantees without strong assumptions on the difficulty function, \(\sigma\). However, this method has been shown empirically to provide good coverage [Nolte]. Several methods exist to construct the difficulty function. One intuitive method is to train an ensemble of regressors, and to use the variance of the ensemble predictions at any point as an estimate of the prediction difficulty. This idea has close ties to the method of deep ensembles, where we assume that if models with different training trajectories provide vastly different outputs for a given input, then the uncertainty of the overall ensemble (the mean of their predictions) should also be large.

To fit with the conformal prediction methodology, we must split the training dataset into a training and calibration set in order to use this method. In contrast to conformal K-fold quantile regression, we cannot use the K-fold training split here to achieve high statistical efficiency because in order to evaluate the normalized residual, we must calculate \(\sigma\) for each point in the calibration set, which must be representative of the ensemble variance of a test point. Therefore, before beginning training, we split the data into a training and calibration set (typically an 80-20 split), then train each of \(K\) regressors on the training set. For each point in the calibration set and each regressor \(k\), we evaluate \(\mu_{\theta_k}\). We take the ensemble prediction \(\mu_*\) to be the mean of the individual predictions, and \(\sigma(X_{cal_i})\) to be the variance of the individual predictions. We then construct the normalized residuals for each point in the calibration set and store these values. To perform inference, we use the same methodology as on the calibration set to evaluate \(\mu_*(X_{te_i})\), and \(\sigma(X_{te_i})\), and construct the conformal interval as above.

Normalized conformal prediction is simple to implement, and can provide adaptive intervals satisfying empirical coverage, but suffers from low statistical efficiency as the K-fold methodology cannot be used. Therefore, this method will likely perform poorly on datasets smaller than a few hundred points.

[Gal]: https://arxiv.org/abs/1506.02142
[Gardner]: https://arxiv.org/abs/1809.11165
[Lakshminarayanan]: https://arxiv.org/abs/1612.01474
[Rasmussen]: http://www.gaussianprocess.org/gpml/
[Vovk2022]: https://arxiv.org/abs/2202.06503
[Angelopoulos2023]: https://arxiv.org/abs/2304.00622
[Romano2019]: https://arxiv.org/abs/1905.03222
[barber2020]: https://arxiv.org/abs/1905.02928
[Nolte]: https://arxiv.org/abs/2202.05265
