import numpy as np
from scipy import integrate, special


def sphere_area(dim):
    """
    Surface area |S^dim| of the dim-dimensional unit sphere.
    """
    return 2 * np.pi ** ((dim + 1) / 2) / special.gamma((dim + 1) / 2)


def epsilon_dip(beta, d):
    """
    Dimension-dependent (local / dipole) threshold:
        ε_dip(d,β) = (|S^{d-2}| / (β |S^{d-1}|))
                     * ∫_{-1}^1 u e^{βu} (1-u^2)^{(d-3)/2} du
    """
    if d < 2:
        raise ValueError("Need d >= 2 (sphere S^{d-1}).")

    Sd_1 = sphere_area(d - 1)
    Sd_2 = sphere_area(d - 2)

    def integrand(u):
        return u * np.exp(beta * u) * (1 - u**2) ** ((d - 3) / 2)

    integral, _ = integrate.quad(integrand, -1, 1, limit=400)

    return (Sd_2 / (beta * Sd_1)) * integral


if __name__ == "__main__":
    for beta in [0.1, 1.0, 2.0, 3.0, 5.0]:
        for d in [2, 3, 5, 10, 100]:
            eps = epsilon_dip(beta, d)
            print(f"beta={beta:>4}, d={d:>2}  -->  epsilon_dip = {eps:.6g}")
