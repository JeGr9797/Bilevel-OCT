# Bilevel-OCT
This repository contains the implementation of a bilevel optimization framework for hyperparameter estimation in classification trees. Unlike standard approaches that rely on grid search, the proposed methodology integrates hyperparameter selection directly into the training process through a hierarchical optimization model.

At the upper level, the model determines the tree hyperparameters, such as the number of active splits and the minimum number of observations per leaf, while at the lower level it solves the classification tree training problem. This bilevel structure allows both tasks to be addressed simultaneously, leading to a more principled and optimization-driven approach to model selection.

The resulting formulation is solved using a Branch-and-Bound algorithm with lazy constraints, following a high-point relaxation logic to enforce the optimality of the lower-level problem. In this way, the repository provides not only a training framework for optimization-based classification trees, but also a computational tool for studying hyperparameter stability, predictive performance, and model interpretability.

In addition to the bilevel tree model, the repository includes utilities for repeated resampling experiments, bagging-inspired evaluation procedures, and frequency-based analysis of selected hyperparameter configurations across multiple runs and datasets. This makes the code especially useful for researchers interested in interpretable machine learning, mathematical optimization, and exact approaches to hyperparameter tuning.

A key advantage of the framework is that hyperparameters are selected while explicitly penalizing tree complexity, often yielding simpler and more interpretable trees without sacrificing predictive accuracy.
