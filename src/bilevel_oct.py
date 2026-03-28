"""
Bilevel Optimal Classification Tree (B-OCT)
===========================================

This module implements a bilevel Optimal Classification Tree (OCT)
classifier using mathematical solver Gurobi. The upper-level variables control tree complexity
(number of active splits and minimum number of observations per active leaf) and the accuracy evaluation,
while the lower-level problem optimizes the training.

It also includes utilities to:
- apply a bagginng-inspired procedure where multiple sampled training/evaluation partitions are generated to
  assess predictive stability and identify robust hyperparameter.
- summarize and visualize the most common hyperparameter pairs.

Author: José Emmanuel Gómez-Rocha
Institution: Tecnológico de Monterrey
contact. emmanuel.gr@tec.mx
------

Notes
-----
- This implementation assumes a binary-tree indexing scheme:
    root = 1
    left child  = 2 * t
    right child = 2 * t + 1
- Leaf and branch node indexing are derived from `max_depth`.
- Gurobi solver is required, due the size of the models, academic or comercial licenses are required.
- More info regarding the licences are found in the Gurobi website.
- The author acknowdledge the contribution from Bo Lin and Bo Tang in the repo "Optimal_Classification_Trees"
  which helped in the development of this code (https://github.com/LucasBoTang/Optimal_Classification_Trees/tree/main).
  The author also acknowdledge the help provided by Eduardo Salazar from AMPL in the first stages of this research.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# =============================================================================
# Data utilities
# =============================================================================
def bagging_sampling(
    path: str | Path,
    frac: float = 0.30,
    seed: int = 67,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a random sample of rows from a CSV file without loading the full file (Bootstrapping-Based Scaling Procedure)

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    frac : float, default=0.30
        Fraction of rows to keep, approximately.
    seed : int, default=67
        Random seed used for row subsampling.
    **kwargs : Any
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame.

    Notes
    -----
    The header row is always preserved (it´s assumed to be the features id).
    """
    if not (0 < frac <= 1):
        raise ValueError("`frac` must satisfy 0 < frac <= 1.")

    rng = random.Random(seed)

    return pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and rng.random() > frac,
        **kwargs,
    )


# =============================================================================
# Model
# =============================================================================
class OptimalDecisionTreeClassifier:
    """
    Bilevel Optimal Classification Tree classifier.

    This model is solved using a Branch-and-Bound algorithm based on the High-Point Relaxation Problem.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum tree depth allowed.
    alpha : float, default=0.1
        Complexity penalty weight.
    warmstart : bool, default=True
        Reserved flag for warm-start logic.
    timelimit : int, default=600
        Time limit for the bilevel optimization model, in seconds.
    output : bool, default=True
        Whether to print Gurobi output and model summaries.

    Attributes
    ----------
    trained : bool
        Whether the model has been fitted.
    optgap : float or None
        Final MIP gap.
    C_used : float
        Number of active splits selected by the model.
    N_min : float
        Minimum observations per active leaf selected by the model.
    model : gp.Model
        Gurobi master model.
    """

    def __init__(
        self,
        max_depth: int = 3,
        alpha: float = 0.1,
        warmstart: bool = True,
        timelimit: int = 600,
        output: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.alpha = alpha
        self.warmstart = warmstart
        self.timelimit = timelimit
        self.output = output

        self.trained: bool = False
        self.optgap: Optional[float] = None

        # Binary heap indexing for nodes
        self.n_index = [i + 1 for i in range(2 ** (self.max_depth + 1) - 1)]
        self.b_index = self.n_index[:-2 ** self.max_depth]   # branch nodes
        self.l_index = self.n_index[-2 ** self.max_depth:]   # leaf nodes

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _cal_baseline(y: np.ndarray) -> int:
        """
        Compute the baseline number of correctly classified observations
        using the majority class.

        Parameters
        ----------
        y : np.ndarray
            Target labels.

        Returns
        -------
        int
            Count of observations in the majority class.
        """
        mode = stats.mode(y, keepdims=False)
        mode_label = mode.mode if hasattr(mode, "mode") else mode[0]
        return int(np.sum(y == mode_label))

    @staticmethod
    def _cal_min_dist(x: np.ndarray) -> list[float]:
        """
        Compute the minimum non-zero spacing per feature.

        Parameters
        ----------
        x : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        list[float]
            Minimum distance per feature.
        """
        min_dis = []

        for j in range(x.shape[1]):
            xj = np.unique(x[:, j])
            xj = np.sort(xj)[::-1]  # descending

            dis = [1.0]
            for i in range(len(xj) - 1):
                dis.append(float(xj[i] - xj[i + 1]))

            md = np.min(dis)
            min_dis.append(md if md != 0 else 1.0)

        return min_dis

    @staticmethod
    def _assert_disjoint_rows(
        x_train_raw: Any,
        x_eval_raw: Any,
    ) -> None:
        """
        Check that train and evaluation sets do not share pandas indices.

        Parameters
        ----------
        x_train_raw : Any
            Training data before conversion to NumPy.
        x_eval_raw : Any
            Evaluation data before conversion to NumPy.

        Raises
        ------
        ValueError
            If overlapping pandas indices are found.
        """
        if hasattr(x_train_raw, "index") and hasattr(x_eval_raw, "index"):
            overlap = x_train_raw.index.intersection(x_eval_raw.index)
            if len(overlap) > 0:
                raise ValueError(
                    "x_eval shares row indices with x_train. "
                    "Please verify the train/evaluation split."
                )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fit(
        self,
        x: Any,
        y: Any,
        x_eval: Optional[Any] = None,
        y_eval: Optional[Any] = None,
    ) -> "OptimalDecisionTreeClassifier":
        """
        Fit the OCT model.

        Parameters
        ----------
        x : array-like
            Training features of shape (n_samples, n_features).
        y : array-like
            Training labels of shape (n_samples,).
        x_eval : array-like, optional
            Evaluation features.
        y_eval : array-like, optional
            Evaluation labels.

        Returns
        -------
        OptimalDecisionTreeClassifier
            Fitted classifier.
        """
        # Keep raw objects for index-disjointness checks
        x_raw = x
        x_eval_raw = x_eval

        if (x_eval is None) ^ (y_eval is None):
            raise ValueError(
                "If evaluation data is provided, both `x_eval` and `y_eval` "
                "must be supplied."
            )

        if x_eval is not None:
            self._assert_disjoint_rows(x_raw, x_eval_raw)

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != 2:
            raise ValueError("`x` must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("`y` must be a 1D array with length equal to x.shape[0].")

        if x_eval is not None and y_eval is not None:
            x_eval = np.asarray(x_eval)
            y_eval = np.asarray(y_eval)

            if x_eval.ndim != 2 or x_eval.shape[1] != x.shape[1]:
                raise ValueError(
                    "`x_eval` must be 2D with the same number of columns as `x`."
                )
            if y_eval.ndim != 1 or y_eval.shape[0] != x_eval.shape[0]:
                raise ValueError(
                    "`y_eval` must be 1D with length equal to x_eval.shape[0]."
                )

        self.n, self.p = x.shape
        self.labels = np.unique(y)

        if self.output:
            print(f"Training data: {self.n} instances, {self.p} features.")

        # Feature scaling
        self.scales = np.max(x, axis=0).astype(float)
        self.scales[self.scales == 0] = 1.0

        x_train_scaled = x / self.scales
        x_eval_scaled = x_eval / self.scales if x_eval is not None else None

        # Build and solve master problem
        model, vars_dict, c_used_var, n_min_var = self._build_mip(
            x_train_scaled,
            y,
            x_eval=x_eval_scaled,
            y_eval=y_eval,
        )
        self.model = model

        callback = self._make_bilevel_callback(
            x_train_scaled,
            y,
            c_used_var,
            n_min_var,
            vars_dict["L"],
        )
        model.optimize(callback)

        try:
            self.optgap = model.MIPGap
        except Exception:
            self.optgap = None

        # Store learned tree
        self._a = {(j, t): vars_dict["a"][j, t].X for (j, t) in vars_dict["a"].keys()}
        self._b = {t: vars_dict["b"][t].X for t in vars_dict["b"].keys()}
        self._c = {(k, t): vars_dict["c"][k, t].X for (k, t) in vars_dict["c"].keys()}
        self._d = {t: vars_dict["d"][t].X for t in vars_dict["d"].keys()}
        self.trained = True

        self.C_used = float(getattr(c_used_var, "X", c_used_var.x))
        self.N_min = float(getattr(n_min_var, "X", n_min_var.x))

        # Objective breakdown
        obj1_val = float(sum(vars_dict["L"][t].X for t in self.l_index) / len(self.l_index))
        obj2_val = float(self.alpha * (self.C_used + self.N_min))

        wE = vars_dict.get("wE", None)
        n_eval = len(x_eval) if x_eval is not None else None

        if wE is not None and n_eval is not None and n_eval > 0:
            obj3_val = 1.0 - float(sum(var.X for var in wE.values()) / n_eval)
        else:
            obj3_val = float("nan")

        if self.output:
            print(f"C_used: {self.C_used:.0f}")
            print(f"N_min:  {self.N_min:.0f}")

            if np.isfinite(obj3_val):
                print(
                    f"ACC_T={obj1_val:.6f}, "
                    f"Complex={obj2_val:.6f}, "
                    f"ACC_E={obj3_val:.6f}"
                )
            else:
                print(
                    f"obj1={obj1_val:.6f}, "
                    f"obj2={obj2_val:.6f}, "
                    "obj3=N/A"
                )

        if hasattr(self, "_EvalVal"):
            try:
                val = float(self._EvalVal.X)
            except Exception:
                val = float(getattr(self._EvalVal, "x", self._EvalVal))
            print(f"Evaluation value (eval errors): {val:.6f}")

        return self

    def predict(self, x: Any) -> np.ndarray:
        """
        Predict labels using the learned OCT.

        Parameters
        ----------
        x : array-like
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self.trained:
            raise RuntimeError("This OptimalDecisionTreeClassifier instance is not fitted yet.")

        x = np.asarray(x)

        labelmap: Dict[int, Any] = {}
        for t in self.l_index:
            for k in self.labels:
                if self._c.get((k, t), 0.0) >= 1e-2:
                    labelmap[t] = k
                    break

        y_pred = []
        for xi in x / self.scales:
            t = 1
            while t not in self.l_index:
                s = sum(self._a.get((j, t), 0.0) * xi[j] for j in range(self.p))
                go_right = s + 1e-9 >= self._b.get(t, 0.0)
                t = 2 * t + 1 if go_right else 2 * t

            y_pred.append(labelmap[t])

        return np.asarray(y_pred)

    # -------------------------------------------------------------------------
    # MIP construction
    # -------------------------------------------------------------------------
    def _build_mip(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_eval: Optional[np.ndarray] = None,
        y_eval: Optional[np.ndarray] = None,
    ) -> Tuple[gp.Model, Dict[str, Any], gp.Var, gp.Var]:
        """
        Build the upper level MIP model.

        Returns
        -------
        model : gp.Model
            Master model.
        vars_dict : dict
            Dictionary with model variables.
        C_used : gp.Var
            Integer variable controlling active splits.
        N_min : gp.Var
            Integer variable controlling minimum observations per leaf.
        """
        model = gp.Model("oct_leader")
        model.Params.OutputFlag = int(self.output)
        model.Params.LogToConsole = int(self.output)
        model.Params.TimeLimit = self.timelimit
        ### Required solve the problem ###
        model.Params.LazyConstraints = 1
        ### Optional, enforces Gurobi to find more incumbent solutions ###
        model.Params.MIPFocus = 1
        ### We noted that clique cuts helps the optimization solver to find better solutions ###
        model.Params.CliqueCuts = 2
        model.ModelSense = GRB.MINIMIZE

        self.n, self.p = x.shape
        min_dis = self._cal_min_dist(x)
        big_m = 1.0 + np.max(min_dis)

        # Leader variables
        C_used = model.addVar(vtype=GRB.INTEGER, lb=1, name="C_used")
        N_min = model.addVar(vtype=GRB.INTEGER, lb=1, name="N_min")

        # Follower variables
        a = model.addVars(self.p, self.b_index, vtype=GRB.BINARY, name="a")
        b = model.addVars(self.b_index, vtype=GRB.CONTINUOUS, name="b")
        c = model.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name="c")
        d = model.addVars(self.b_index, vtype=GRB.BINARY, name="d")
        z = model.addVars(self.n, self.l_index, vtype=GRB.BINARY, name="z")
        l = model.addVars(self.l_index, vtype=GRB.BINARY, name="l")
        L = model.addVars(self.l_index, vtype=GRB.CONTINUOUS, name="L")
        M = model.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name="M")
        N = model.addVars(self.l_index, vtype=GRB.CONTINUOUS, name="N")
        Nbar = model.addVars(self.l_index, vtype=GRB.CONTINUOUS, name="N_bar")

        wE = None
        obj3 = 0.0

        # evaluation block from the upper level model
        if x_eval is not None and y_eval is not None:
            n_eval = x_eval.shape[0]
            zE = model.addVars(n_eval, self.l_index, vtype=GRB.BINARY, name="zE")
            wE = model.addVars(
                n_eval,
                self.labels,
                self.l_index,
                vtype=GRB.BINARY,
                name="wE",
            )

            model.addConstrs(
                (wE[i, k, t] <= zE[i, t]
                 for i in range(n_eval) for t in self.l_index for k in self.labels),
                name="eval_lin_1",
            )
            model.addConstrs(
                (wE[i, k, t] <= c[k, t]
                 for i in range(n_eval) for t in self.l_index for k in self.labels),
                name="eval_lin_2",
            )
            model.addConstrs(
                (wE[i, k, t] >= zE[i, t] + c[k, t] - 1
                 for i in range(n_eval) for t in self.l_index for k in self.labels),
                name="eval_lin_3",
            )

            model.addConstrs((zE.sum(i, "*") == 1 for i in range(n_eval)), name="eval_assign")
            model.addConstrs(
                (zE[i, t] <= l[t] for t in self.l_index for i in range(n_eval)),
                name="eval_leaf_active",
            )

            for t in self.l_index:
                left = (t % 2 == 0)
                ta = t // 2

                while ta != 0:
                    if left:
                        model.addConstrs(
                            (
                                gp.quicksum(
                                    a[j, ta] * (x_eval[i, j] + min_dis[j])
                                    for j in range(self.p)
                                )
                                + big_m * (1 - d[ta])
                                <= b[ta] + big_m * (1 - zE[i, t])
                                for i in range(n_eval)
                            ),
                            name=f"eval_left_t{t}_a{ta}",
                        )
                    else:
                        model.addConstrs(
                            (
                                gp.quicksum(a[j, ta] * x_eval[i, j] for j in range(self.p))
                                >= b[ta] - (1 - zE[i, t])
                                for i in range(n_eval)
                            ),
                            name=f"eval_right_t{t}_a{ta}",
                        )

                    left = (ta % 2 == 0)
                    ta //= 2

            # Evaluation accuracy
            obj3 = 1 - gp.quicksum(
                wE[i, k, t] * (y_eval[i] == k)
                for i in range(n_eval)
                for t in self.l_index
                for k in self.labels
            ) / n_eval

        # Leader-follower links
        model.addConstr(d.sum() == C_used, name="complexity_exact")
        model.addConstr(C_used <= 2 ** (self.max_depth + 1) - 1, name="complexity_bound")
        model.addConstr(N_min <= math.floor(0.5 * self.n), name="nmin_upper")
        model.addConstr(N_min >= math.ceil(0.1 * self.n), name="nmin_lower")

        # Training constraints (lower level)
        model.addConstrs(
            (
                L[t] >= N[t] - M[k, t] - self.n * (1 - c[k, t])
                for t in self.l_index for k in self.labels
            ),
            name="loss_lb",
        )
        model.addConstrs(
            (
                L[t] <= N[t] - M[k, t] + self.n * c[k, t]
                for t in self.l_index for k in self.labels
            ),
            name="loss_ub",
        )

        model.addConstrs(
            (
                gp.quicksum((y[i] == k) * z[i, t] for i in range(self.n)) == M[k, t]
                for t in self.l_index for k in self.labels
            ),
            name="class_count",
        )

        model.addConstrs((z.sum("*", t) == N[t] for t in self.l_index), name="leaf_size")
        model.addConstrs((c.sum("*", t) == l[t] for t in self.l_index), name="leaf_label")

        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2

            while ta != 0:
                if left:
                    model.addConstrs(
                        (
                            gp.quicksum(
                                a[j, ta] * (x[i, j] + min_dis[j])
                                for j in range(self.p)
                            )
                            + big_m * (1 - d[ta])
                            <= b[ta] + big_m * (1 - z[i, t])
                            for i in range(self.n)
                        ),
                        name=f"train_left_t{t}_a{ta}",
                    )
                else:
                    model.addConstrs(
                        (
                            gp.quicksum(a[j, ta] * x[i, j] for j in range(self.p))
                            >= b[ta] - (1 - z[i, t])
                            for i in range(self.n)
                        ),
                        name=f"train_right_t{t}_a{ta}",
                    )

                left = (ta % 2 == 0)
                ta //= 2

        model.addConstrs((z.sum(i, "*") == 1 for i in range(self.n)), name="assign_one_leaf")
        model.addConstrs(
            (z[i, t] <= l[t] for t in self.l_index for i in range(self.n)),
            name="assign_active_leaf",
        )

        model.addConstrs((a.sum("*", t) == d[t] for t in self.b_index), name="split_activation")
        model.addConstrs((b[t] <= d[t] for t in self.b_index), name="threshold_activation")
        model.addConstrs(
            (d[t] <= d[t // 2] for t in self.b_index if t != 1),
            name="tree_consistency",
        )

        model.addConstrs((z.sum("*", t) >= Nbar[t] for t in self.l_index), name="nbar_lb")
        model.addConstrs((Nbar[t] <= self.n * l[t] for t in self.l_index), name="nbar_leaf")
        model.addConstrs((Nbar[t] <= N_min for t in self.l_index), name="nbar_nmin_ub")
        model.addConstrs(
            (Nbar[t] >= N_min - self.n * (1 - l[t]) for t in self.l_index),
            name="nbar_nmin_lb",
        )

        # Objective from the training in the lower lovel #
        obj1 = gp.quicksum(L[t] for t in self.l_index) / len(self.l_index)

        # Tree complexity (standardized) #
        obj2 = self.alpha * (
            C_used / (2 ** (self.max_depth + 1) - 1)
            - (
                (N_min - math.ceil(0.1 * self.n))
                / (math.floor(0.5 * self.n) - math.ceil(0.1 * self.n))
            )
        )

        #model.setObjective(obj1 + obj2 + obj3)
        model.setObjective(obj2 + obj3)
      

        vars_dict = {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "z": z,
            "l": l,
            "L": L,
            "M": M,
            "N": N,
            "Nbar": Nbar,
            "wE": wE,
        }

        return model, vars_dict, C_used, N_min

    def _build_follower_sub(
        self,
        x: np.ndarray,
        y: np.ndarray,
        C_fix: int,
        Nmin_fix: int,
    ) -> gp.Model:
        """
        Build follower subproblem (training) for a fixed pair (C_fix, Nmin_fix).
        """
        n, p = self.n, self.p
        min_dis = self._cal_min_dist(x)
        big_m = 1.0 + np.max(min_dis)

        mf = gp.Model("oct_follower")
        # Hide the log from each evaluation of the follower #
        mf.Params.OutputFlag = 0

        a = mf.addVars(p, self.b_index, vtype=GRB.BINARY, name="a")
        b = mf.addVars(self.b_index, vtype=GRB.CONTINUOUS, name="b")
        c = mf.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name="c")
        d = mf.addVars(self.b_index, vtype=GRB.BINARY, name="d")
        z = mf.addVars(n, self.l_index, vtype=GRB.BINARY, name="z")
        l = mf.addVars(self.l_index, vtype=GRB.BINARY, name="l")
        L = mf.addVars(self.l_index, vtype=GRB.CONTINUOUS, name="L")
        M = mf.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name="M")
        N = mf.addVars(self.l_index, vtype=GRB.CONTINUOUS, name="N")

        mf.addConstrs(
            (L[t] >= N[t] - M[k, t] - n * (1 - c[k, t]) for t in self.l_index for k in self.labels)
        )
        mf.addConstrs(
            (L[t] <= N[t] - M[k, t] + n * c[k, t] for t in self.l_index for k in self.labels)
        )

        mf.addConstrs(
            (
                gp.quicksum((y[i] == k) * z[i, t] for i in range(n)) == M[k, t]
                for t in self.l_index for k in self.labels
            )
        )
        mf.addConstrs((z.sum("*", t) == N[t] for t in self.l_index))
        mf.addConstrs((c.sum("*", t) == l[t] for t in self.l_index))

        for t in self.l_index:
            left = (t % 2 == 0)
            ta = t // 2

            while ta != 0:
                if left:
                    mf.addConstrs(
                        (
                            gp.quicksum(a[j, ta] * (x[i, j] + min_dis[j]) for j in range(p))
                            + big_m * (1 - d[ta])
                            <= b[ta] + big_m * (1 - z[i, t])
                            for i in range(n)
                        )
                    )
                else:
                    mf.addConstrs(
                        (
                            gp.quicksum(a[j, ta] * x[i, j] for j in range(p))
                            >= b[ta] - (1 - z[i, t])
                            for i in range(n)
                        )
                    )

                left = (ta % 2 == 0)
                ta //= 2

        mf.addConstrs((z.sum(i, "*") == 1 for i in range(n)))
        mf.addConstrs((z[i, t] <= l[t] for t in self.l_index for i in range(n)))

        mf.addConstrs((a.sum("*", t) == d[t] for t in self.b_index))
        mf.addConstrs((b[t] <= d[t] for t in self.b_index))
        mf.addConstrs((d[t] <= d[t // 2] for t in self.b_index if t != 1))

        mf.addConstr(d.sum() == C_fix, name="fix_C")
        mf.addConstrs((z.sum("*", t) >= l[t] * Nmin_fix for t in self.l_index), name="fix_Nmin")

        mf.setObjective(
            gp.quicksum(L[t] for t in self.l_index) / len(self.l_index),
            GRB.MINIMIZE,
        )
        return mf

    def _make_bilevel_callback(
        self,
        x: np.ndarray,
        y: np.ndarray,
        C_var: gp.Var,
        Nmin_var: gp.Var,
        L_vars_dict: gp.tupledict,
    ) -> Callable[[gp.Model, int], None]:
        """
        Create lazy-constraint callback enforcing bilevel feasible solutions.
        """
        eps = 1e-6
        idxL = list(self.l_index)

        def callback(model: gp.Model, where: int) -> None:
            if where == GRB.Callback.MIPSOL:
                solcnt = int(model.cbGet(GRB.Callback.MIPSOL_SOLCNT))

                # Skip the very first incumbent
                if solcnt <= 1:
                    return

                C_val = model.cbGetSolution(C_var)
                Nmin_val = model.cbGetSolution(Nmin_var)

                C_fix = max(1, int(round(C_val)))
                Nmin_fix = max(1, int(round(Nmin_val)))

                follower = self._build_follower_sub(x, y, C_fix, Nmin_fix)
                follower.optimize()

                if follower.Status != GRB.OPTIMAL:
                    return

                phi = follower.ObjVal
                Lsum_val = sum(model.cbGetSolution(L_vars_dict[t]) for t in idxL)

                if Lsum_val > phi + eps:
                    # If the current solution violates the value function, the bilevel sulution is not feasible, a cut is applied #
                    expr = gp.quicksum(L_vars_dict[t] for t in idxL)
                    model.cbLazy(expr <= phi)

        return callback

# =============================================================================
# Experiment utilities
# =============================================================================
def run_sampling_experiments(
    path_csv: str | Path,
    n_runs: int = 10,
    frac: float = 0.30,
    base_seed: int = 71,
    test_size: float = 0.20,
    max_depth: int = 4,
    alpha: float = 0.2,
    timelimit: int = 600,
    output: bool = False,
) -> pd.DataFrame:
    """
    Run repeated bagging-sampling experiments on a dataset.

    Each run:
    1. Apply the inspired Bootstrapping to the dataset CSV file.
    2. Splits the sample into train/evaluation sets.
    3. Trains the OCT model.
    4. Stores accuracy and optimization statistics.

    Returns
    -------
    pd.DataFrame
        One row per run with solver and prediction metrics.
    """
    rows = []

    for r in range(n_runs):
        seed_sample = base_seed + r
        seed_split = 10_000 + base_seed + r

        t0 = time.time()

        try:
            df = bagging_sampling(path_csv, frac=frac, seed=seed_sample)
            df["class"] = df["class"].replace(-1, 0)

            X = df.drop(columns=["class"]).values
            y = df["class"].values

            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=seed_split,
                # stratify=y, # optional stratify the sampling, can lead in some cases to not useful samplings
            )

            oct_model = OptimalDecisionTreeClassifier(
                max_depth=max_depth,
                alpha=alpha,
                timelimit=timelimit,
                output=output,
            )
            oct_model.fit(X_tr, y_tr, x_eval=X_ev, y_eval=y_ev)

            yhat_tr = oct_model.predict(X_tr)
            yhat_ev = oct_model.predict(X_ev)

            acc_tr = accuracy_score(y_tr, yhat_tr)
            acc_ev = accuracy_score(y_ev, yhat_ev)

            model = getattr(oct_model, "model", None)
            status = getattr(model, "Status", None)
            objval = getattr(model, "ObjVal", np.nan) if model is not None else np.nan
            mipgap = getattr(oct_model, "optgap", np.nan)

            elapsed = time.time() - t0

            rows.append(
                {
                    "run": r,
                    "n_rows_sample": len(df),
                    "n_train": len(y_tr),
                    "n_eval": len(y_ev),
                    "C_used": getattr(oct_model, "C_used", np.nan),
                    "N_min": getattr(oct_model, "N_min", np.nan),
                    "ACC_T": acc_tr,
                    "ACC_E": acc_ev,
                    "ObjVal": objval,
                    "MIPGap": mipgap,
                    "Status": status,
                    "TimeSec": elapsed,
                }
            )

        except Exception as exc:
            elapsed = time.time() - t0
            rows.append(
                {
                    "run": r,
                    "n_rows_sample": np.nan,
                    "n_train": np.nan,
                    "n_eval": np.nan,
                    "C_used": np.nan,
                    "N_min": np.nan,
                    "ACC_T": np.nan,
                    "ACC_E": np.nan,
                    "ObjVal": np.nan,
                    "MIPGap": np.nan,
                    "Status": "ERROR",
                    "TimeSec": elapsed,
                    "Error": repr(exc),
                }
            )

    return pd.DataFrame(rows)


def analyze_runs(
    df_runs: pd.DataFrame,
    top_k: int = 5,
    show_plots: bool = True,
    title_prefix: str = "",
) -> Dict[str, Any]:
    """
    Summarize repeated runs and identify the most common (C_used, N_min) pairs.

    Parameters
    ----------
    df_runs : pd.DataFrame
        DataFrame returned by `run_sampling_experiments`.
    top_k : int, default=5
        Number of top pairs to display.
    show_plots : bool, default=True
        Whether to produce matplotlib plots.
    title_prefix : str, default=""
        Prefix for plot titles.

    Returns
    -------
    dict
        Summary dictionary with descriptive statistics and top pairs.
    """
    cols = ["C_used", "N_min", "ACC_T", "ACC_E", "ObjVal", "MIPGap", "TimeSec"]

    print("\n================ SUMMARY =================")
    summary = df_runs[cols].describe()
    print(summary)

    df_ok = df_runs.dropna(subset=["C_used", "N_min"]).copy()
    if df_ok.empty:
        print("No valid runs available for analysis.")
        return {
            "summary": summary,
            "pair_counts": None,
            "top_pair": None,
            "top_pairs": None,
        }

    df_ok["C_used"] = df_ok["C_used"].round().astype(int)
    df_ok["N_min"] = df_ok["N_min"].round().astype(int)

    if show_plots:
        plt.figure()
        df_ok["C_used"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("C_used")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix}Frequency of C_used".strip())
        plt.tight_layout()
        plt.show()

        plt.figure()
        df_ok["N_min"].value_counts().sort_index().plot(kind="bar")
        plt.xlabel("N_min")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix}Frequency of N_min".strip())
        plt.tight_layout()
        plt.show()

    pair_counts = (
        df_ok.groupby(["C_used", "N_min"])
        .size()
        .reset_index(name="freq")
        .sort_values("freq", ascending=False)
    )

    top_pair = pair_counts.iloc[0]
    print("\n=========== MOST COMMON PAIR ===========")
    print(top_pair)

    if show_plots:
        plt.figure(figsize=(6, 4))
        plt.bar(
            [fr"$(C={top_pair.C_used},\,N_{{\min}}={top_pair.N_min})$"],
            [top_pair.freq],
        )
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix}Most common pair $(C, N_{{\min}})$".strip())
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.show()

    top_pairs = pair_counts.head(top_k)

    labels = [
        fr"$(C={c},\,N_{{\min}}={n})$"
        for c, n in zip(top_pairs["C_used"], top_pairs["N_min"])
    ]

    print(f"\n=========== TOP {top_k} PAIRS ===========")
    print(top_pairs)

    if show_plots:
        plt.figure(figsize=(8, 4.8))
        bars = plt.bar(labels, top_pairs["freq"])

        plt.xlabel(r"Pair $(C, N_{\min})$")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix}Top {top_k} most common pairs".strip())
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()

    return {
        "summary": summary,
        "pair_counts": pair_counts,
        "top_pair": top_pair,
        "top_pairs": top_pairs,
    }


def run_grid_experiments_with_analysis(
    paths_csv: Iterable[str | Path],
    frac_by_dataset: Dict[str, float],
    alphas: Iterable[float] = (0.1, 0.3, 0.5),
    n_runs: int = 30,
    base_seed: int = 71,
    test_size: float = 0.20,
    max_depth: int = 4,
    timelimit: int = 600,
    output: bool = False,
    top_k_pairs: int = 5,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run experiments across multiple datasets and alpha values.

    Parameters
    ----------
    paths_csv : iterable
        CSV paths.
    frac_by_dataset : dict
        Mapping from filename to sampling fraction.
    alphas : iterable, default=(0.1, 0.3, 0.5)
        Alpha values to test.
    n_runs : int, default=30
        Number of repeated runs per dataset/alpha pair.
    base_seed : int, default=71
        Base random seed.
    test_size : float, default=0.20
        Evaluation proportion.
    max_depth : int, default=4
        Maximum tree depth.
    timelimit : int, default=600
        Solver time limit.
    output : bool, default=False
        Whether to show Gurobi logs.
    top_k_pairs : int, default=5
        Number of top (C_used, N_min) pairs to report.
    show_plots : bool, default=True
        Whether to show analysis plots.

    Returns
    -------
    df_all : pd.DataFrame
        Concatenated run-level results.
    summary_df : pd.DataFrame
        Block-level summary results.
    """
    all_blocks = []
    summary_rows = []

    for path in paths_csv:
        path = Path(path)
        dataset_name = path.stem
        fname = path.name

        if fname not in frac_by_dataset:
            raise ValueError(f"No sampling fraction provided for dataset: {fname}")

        frac = frac_by_dataset[fname]
        print(f"\n>>> Dataset {dataset_name} uses frac={frac}")

        for alpha in alphas:
            print("\n" + "=" * 70)
            print(f"DATASET = {dataset_name} | alpha = {alpha} | frac = {frac}")
            print("=" * 70)

            df_runs = run_sampling_experiments(
                path_csv=path,
                n_runs=n_runs,
                frac=frac,
                base_seed=base_seed,
                test_size=test_size,
                max_depth=max_depth,
                alpha=alpha,
                timelimit=timelimit,
                output=output,
            )

            df_runs = df_runs.copy()
            df_runs["dataset"] = dataset_name
            df_runs["alpha"] = alpha
            df_runs["frac"] = frac

            title_prefix = f"{dataset_name} | α={alpha} | frac={frac} | "
            analysis = analyze_runs(
                df_runs=df_runs,
                top_k=top_k_pairs,
                show_plots=show_plots,
                title_prefix=title_prefix,
            )

            df_ok = df_runs.dropna(subset=["ACC_E", "ACC_T", "C_used", "N_min"]).copy()

            row = {
                "dataset": dataset_name,
                "alpha": alpha,
                "frac": frac,
                "runs_total": int(len(df_runs)),
                "runs_ok": int(len(df_ok)),
                "ACC_E_mean": float(df_ok["ACC_E"].mean()) if len(df_ok) else np.nan,
                "ACC_E_std": float(df_ok["ACC_E"].std()) if len(df_ok) else np.nan,
                "ACC_T_mean": float(df_ok["ACC_T"].mean()) if len(df_ok) else np.nan,
                "TimeSec_mean": float(df_ok["TimeSec"].mean()) if len(df_ok) else np.nan,
                "MIPGap_mean": float(df_ok["MIPGap"].mean()) if len(df_ok) else np.nan,
                "C_used_mean": float(df_ok["C_used"].mean()) if len(df_ok) else np.nan,
                "N_min_mean": float(df_ok["N_min"].mean()) if len(df_ok) else np.nan,
            }

            if analysis["top_pair"] is not None:
                top_pair = analysis["top_pair"]
                row["top_pair_C_used"] = int(top_pair["C_used"])
                row["top_pair_N_min"] = int(top_pair["N_min"])
                row["top_pair_freq"] = int(top_pair["freq"])
                row["top_pair_pct"] = float(top_pair["freq"]) / max(1, len(df_ok))
            else:
                row["top_pair_C_used"] = np.nan
                row["top_pair_N_min"] = np.nan
                row["top_pair_freq"] = np.nan
                row["top_pair_pct"] = np.nan

            summary_rows.append(row)
            all_blocks.append(df_runs)

    df_all = pd.concat(all_blocks, ignore_index=True) if all_blocks else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    return df_all, summary_df
