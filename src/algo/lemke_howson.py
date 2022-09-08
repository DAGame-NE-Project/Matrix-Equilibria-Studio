
import warnings
from itertools import cycle

import numpy as np
import numpy.typing as npt
from typing import Tuple, Set, Iterable


from nashpy.integer_pivoting import (
    make_tableau,
    non_basic_variables,
    pivot_tableau,
)


def shift_tableau(tableau: npt.NDArray, shape: Tuple[int, ...]) -> npt.NDArray:
    return np.append(
        np.roll(tableau[:, :-1], shape[0], axis=1),
        np.ones((shape[0], 1)),
        axis=1,
    )


def tableau_to_strategy(
    tableau: npt.NDArray, basic_labels: Set[int], strategy_labels: Iterable
) -> npt.NDArray:
    vertex = []
    for column in strategy_labels:
        if column in basic_labels:
            for i, row in enumerate(tableau[:, column]):
                if row != 0:
                    vertex.append(tableau[i, -1] / row)
        else:
            vertex.append(0)
    strategy = np.array(vertex)
    return strategy / sum(strategy)


def lemke_howson(
    A: npt.NDArray, B: npt.NDArray, initial_dropped_label: int = 0
) -> Tuple[npt.NDArray, npt.NDArray]:
    if np.min(A) <= 0:
        A = A + abs(np.min(A)) + 1
    if np.min(B) <= 0:
        B = B + abs(np.min(B)) + 1

    # build tableaux
    col_tableau = make_tableau(A)
    col_tableau = shift_tableau(col_tableau, A.shape)
    row_tableau = make_tableau(B.transpose())
    full_labels = set(range(sum(A.shape)))

    if initial_dropped_label in non_basic_variables(row_tableau):
        tableux = cycle((row_tableau, col_tableau))
    else:
        tableux = cycle((col_tableau, row_tableau))

    # First pivot (to drop a label)
    entering_label = pivot_tableau(next(tableux), initial_dropped_label)
    while (
        non_basic_variables(row_tableau).union(non_basic_variables(col_tableau))
        != full_labels
    ):
        entering_label = pivot_tableau(next(tableux), next(iter(entering_label)))

    row_strategy = tableau_to_strategy(
        row_tableau, non_basic_variables(col_tableau), range(A.shape[0])
    )
    col_strategy = tableau_to_strategy(
        col_tableau,
        non_basic_variables(row_tableau),
        range(A.shape[0], sum(A.shape)),
    )

    if row_strategy.shape != (A.shape[0],) and col_strategy.shape != (A.shape[0],):
        msg = """The Lemke Howson algorithm has returned probability vectors of incorrect shapes. 
        This indicates an error. Your game could be degenerate."""
        warnings.warn(msg, RuntimeWarning)
    return row_strategy, col_strategy