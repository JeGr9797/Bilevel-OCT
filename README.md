# Bilevel-OCT
This repository contains the implementation of a bilevel optimization framework for hyperparameter estimation in classification trees, namely Bilevel Optimal Classification Trees (B-OCT).

## Related paper

This repository contains the main code accompanying the paper **“A New Bilevel Approach for Hyperparameter Estimation in Optimal Classification Trees”** by **José-Fernando Camacho-Vallejo, José-Emmanuel Gómez-Rocha, and Justo Puerto**. Unlike standard approaches based on grid search, the proposed methodology integrates hyperparameter selection directly into the training process through a hierarchical optimization model. For the computational experiments, we implement a generic Branch-and-Bound algorithm for the bilevel problem, based on branching on the High-Point Relaxation (HPR) problem and verifying bilevel feasibility whenever an integer solution is found. Although the methodology is exact and yields optimal solutions, its performance is naturally affected by the computational complexity of the underlying optimization problems. To improve scalability on larger datasets, the repository also includes a bagging-inspired resampling strategy.

## Brief description

At the upper level, the model determines the tree hyperparameters, such as the number of active splits and the minimum number of observations per leaf, while at the lower level it solves the classification tree training problem. This bilevel structure allows both tasks to be addressed simultaneously, leading to a more principled and optimization-driven approach to model selection.

The resulting formulation is solved using a Branch-and-Bound algorithm with lazy constraints, following a high-point relaxation logic to enforce the optimality of the lower-level problem. In this way, the repository provides not only a training framework for optimization-based classification trees, but also a computational tool for studying hyperparameter stability, predictive performance, and model interpretability.

In addition to the bilevel tree model, the repository includes utilities for repeated resampling experiments, bagging-inspired evaluation procedures, and frequency-based analysis of selected hyperparameter configurations across multiple runs and datasets. This makes the code especially useful for researchers interested in interpretable machine learning, mathematical optimization, and exact approaches to hyperparameter tuning.

A key advantage of the framework is that hyperparameters are selected while explicitly penalizing tree complexity, often yielding simpler and more interpretable trees without sacrificing predictive accuracy.

---

## Overview

Hyperparameter tuning is a crucial step in the construction of classification trees. Standard approaches typically rely on grid search, which may be computationally inefficient and limited by the predefined set of candidate values. This repository provides an optimization-based alternative in which hyperparameter selection is embedded directly into the learning problem.

The main idea is to formulate the problem as a bilevel optimization model:

- **Upper level:** minimizes evaluation accuracy of the classification tree and selects the hyperparameters controlling tree complexity.
- **Lower level:** solves the training problem of the classification tree.

This hierarchical formulation makes it possible to jointly determine the tree structure and its complexity in a unified optimization framework.

In addition to the exact optimization model, the repository includes an experimental pipeline based on repeated resampling. A **bagging-inspired framework** is used where s sampled training/evaluation splits are generated to study the stability of the selected hyperparameters and the predictive behavior of the learned trees.

---

## Methodology

The repository is built around three main components:

### 1. Bilevel OCT model
The core model is a bilevel-inspired Optimal Classification Tree (OCT) formulation in which:

- the **upper-level variables** determine hyperparameters such as:
  - the number of active splits in the tree,
  - the minimum number of observations per active leaf;
  - Aditionally minimize misclassfications in the evaluation.

- the **lower-level problem** determines:
  - the feature used at each branch node,
  - the split thresholds,
  - the assignment of observations to leaves,
  - the class prediction of each leaf.

This structure integrates hyperparameter tuning directly into the optimization process.

### 2. Ad-hoc Branch-and-Bound algorithm for B-OCT
The resulting model is solved using a **generic Branch-and-Bound algorithm for bilevel optimization**. The solution approach is based on branching on the **High-Point Relaxation (HPR)** problem and verifying **bilevel feasibility** whenever an integer solution is found.

To account for the hierarchical nature of the formulation, the algorithm checks whether the lower-level problem is optimally solved for the current upper-level hyperparameter values. If bilevel feasibility is violated, **lazy constraints** are added to exclude the incumbent solution and enforce consistency with the lower-level optimal response.

In this way, the procedure remains exact and is able to recover the optimal solution of the bilevel model, although its computational performance is naturally affected by the complexity of the underlying optimization problems.

### 3. Bagging-inspired repeated resampling
The repository also includes a repeated resampling experimental pipeline. Across multiple runs:

- a random sample of the dataset is extracted,
- the data is split into training and evaluation sets,
- the bilevel OCT model is trained,
- the selected hyperparameters and predictive metrics are recorded.

This procedure is **bagging-inspired** because it generates multiple sampled versions of the training/evaluation process in order to assess:

- predictive stability,
- robustness of the selected hyperparameters,
- frequency of the most common hyperparameter pairs across runs.

## Requirements

To run the code in this repository, the following Python libraries are required:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `gurobipy`
