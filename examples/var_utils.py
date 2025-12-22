import numpy as np

from srvar.var import design_matrix, is_stationary


def main() -> None:
    y = np.arange(20, dtype=float).reshape(10, 2)
    x, yt = design_matrix(y, p=2, include_intercept=True)
    print("design_matrix shapes:", x.shape, yt.shape)

    beta = np.zeros((1 + 2 * 2, 2), dtype=float)
    beta[1, 0] = 0.5
    beta[2, 1] = 0.5
    print("is_stationary:", is_stationary(beta, n=2, p=2))


if __name__ == "__main__":
    main()
