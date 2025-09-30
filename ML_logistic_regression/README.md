# Logistic Regression Demonstrations
## Introduction

This directory is to demonstrate logistic regression. Essential concepts and mathematics is
shown here. A quick `sklearn` implementation and a demo of implementing logistic regression 
from scratch in `NumPy`, training model with gradient descent, are shown in the notebook.

## Mathematics 
For a sample with $K$ features, $x_1,\dots, x_K$ and binary label $y\in\{0,1\}$,
the model with parameters $\beta_0,\dots,\beta_K$ outputs the probability of $y=1$ as

$$
p=\frac{\exp(\sum_{j=1}^n{\beta_jx_j+\beta_0)}}{1+\exp(\sum_{j=1}^n{\beta_jx_j+\beta_0)})}.
$$

Denote $\mathbf{x}=[1,x_1,\dots,x_K]^T\in \mathbb{R}^{K+1}$, $\mathbf{\beta}=[\beta_0,\dots,\beta_K]^T\in \mathbb{R}^{K+1}$, and sigmoid function

$$
\sigma(x) = \frac{\exp(x)}{1+\exp(x)},
$$

we have

$$
p = f(\mathbf{x};\mathbf{\beta}) = \sigma(\mathbf{\beta}^T\mathbf{x}),
$$

and for each record $i$,

$$
p_i = f(\mathbf{x}_i;\mathbf{\beta}).
$$

For $n$ samples, we have the negative log-likelihood as loss fucntion

$$
l(\mathbf{\beta}) = -\frac{1}{n}\sum_{i=1}^n \left[y_i\ln p_i + (1-y_i)\ln(1-p_i)\right].
$$

Train the model by minimizing the loss function with respect to the parameters as

$$
\hat{\mathbf{\beta}} = \arg\min_{\mathbf{\beta}} l(\mathbf{\beta}).
$$

Using gradient descent algorithm to minimize the loss. With learning rate $\lambda$, in each iteration,

$$
\mathbf{\beta} \leftarrow \hat{\mathbf{\beta}} - \lambda \frac{\partial}{\partial\mathbf{\beta}}  l(\mathbf{\beta}).
$$

The loss contribution of record $i$
is

$$
l_i(\mathbf{\beta}) = -\left[y_i\ln p_i + (1-y_i)\ln(1-p_i)\right].
$$

Denote the linear combination of features as $h_i = \mathbf{\beta}^T\mathbf{x}_i$, the loss gradient is

$$
\frac{\partial}{\partial \beta} l_i(\mathbf{\beta}) = \frac{\partial h_i}{\partial\mathbf{\beta}} \frac{\partial p_i}{\partial h_i}  \frac{\partial l_i(\mathbf{\beta})} {\partial p_i},
$$

where

$$
\frac{\partial h_i}{\partial\mathbf{\beta}} = \mathbf{x}_i, \frac{\partial p_i}{\partial h} = p_i(1-p_i), \frac{\partial l_i(\mathbf{\beta})}{\partial p_i} = \frac{p_i-y_i}{p_i(1-p_i)}.
$$

So we have

$$
\frac{\partial }{\partial \beta} l_i(\mathbf{\beta})= \mathbf{x}_i(p_i-y_i),
$$

and

$$
\frac{\partial}{\partial\mathbf{\beta}}  l(\mathbf{\beta}) = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i(p_i-y_i).
$$

## Notebooks
Demonstrations of logistic regression with sklearn and a `Numpy` implementation are shown in
are in [**demo_logistic_regression.ipynb**](https://github.com/liyunfan2012/demo-AI-ML/blob/main/ML_logistic_regression/demo_logistic_regression.ipynb) 
