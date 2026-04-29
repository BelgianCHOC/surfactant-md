import streamlit as st
import numpy as np
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt

T = 298.15          # temperature in Kelvin
kB = 1.380649e-23   # Boltzmann constant, J/K
R  = 8.314          # gas constant, J/(mol·K)

kBT_eff = kB * T * 1e18 * 1e3  # ≈ 4.116  mN/m per nm⁻²
DGads_kBT = -44050.0 / (R * T)  # ≈ -17.77, dimensionless
gamma0 = 70.66   # mN/m — pure water ST (Γ = 0 row in Table S4)
X_CMC  = 1.477e-6  # mole fraction
Gamma_all = np.array([0.00, 0.28, 0.56, 0.83, 1.11,
                       1.39, 1.67, 1.94, 2.22, 2.50, 2.78])  # nm⁻²
ST_all    = np.array([70.66, 69.36, 65.59, 59.30, 54.41,
                       48.60, 44.29, 38.70, 32.18, 25.69, 17.15])  # mN/m

def eq5_ST(Gamma, r, B):
    Z = Gamma * np.pi * r**2
    return gamma0 - kBT_eff * (Gamma / (1.0 - Z)**2 + B * Gamma**2)

def eq6_lnX(Gamma, r, B):
    Z = Gamma * np.pi * r**2
    return (DGads_kBT
            + np.log(Z / (1.0 - Z))
            + (3*Z - 2*Z**2) / (1.0 - Z)**2
            + 2*B*Gamma)

# iterative loop function

#Type the function signature and the loop setup

def run_iterations(Gamma_max_init, tol=1e-4, max_iter=50):
    Gamma_max = Gamma_max_init
    history = []

# data filtering step

    for i in range(max_iter):
        mask = Gamma_all <= Gamma_max + 1e-9
        G_fit = Gamma_all[mask]
        S_fit = ST_all[mask]


# curve_fit step

        try:
            popt, _ = curve_fit(
                eq5_ST, G_fit, S_fit,
                p0=[0.3, -0.1],
                bounds=([1e-6, -np.inf], [np.inf, 0.0])
            )
            r, B = popt
        except RuntimeError:
            return None, None, None, history

        # solve Equation 6 for Γ at X = CMC

        lnX_CMC = np.log(X_CMC)
        try:
            Gamma_new = brentq(
                lambda G: eq6_lnX(G, r, B) - lnX_CMC,
                1e-6, 2.78,
                xtol=1e-8
            )
        except ValueError:
            return None, None, None, history

# save to history and check convergence

        history.append({
            "Iteration": i + 1,
            "Gamma_max used": round(Gamma_max, 4),
            "r (nm)": round(r, 4),
            "B (nm²)": round(B, 4),
            "Gamma_CMC": round(Gamma_new, 4),
        })

        if abs(Gamma_new - Gamma_max) < tol:
            return r, B, Gamma_new, history

        Gamma_max = Gamma_new

    return r, B, Gamma_max, history



st.title("C12E6 Adsorption Isotherm — MD-MTT Framework")
st.sidebar.header("Settings")

Gamma_init = st.sidebar.slider(
    "Initial Γ_max guess (nm⁻²)",
    min_value=0.28, max_value=2.78, value=1.50, step=0.01
)
tol = st.sidebar.number_input(
    "Convergence tolerance (nm⁻²)", value=5e-1, format="%.2e"
)
run = st.sidebar.button("Run")

if run:
    r, B, Gamma_CMC, history = run_iterations(Gamma_init, tol=float(tol))

    if r is None:
        st.error("Iteration failed. Try a different initial guess.")
    else:
        ST_CMC = eq5_ST(Gamma_CMC, r, B)

        st.subheader("Convergence history")
        st.dataframe(history)

        st.markdown(f"**Converged:** r = {r:.4f} nm,  B = {B:.4f} nm²")
        st.markdown(f"**Predicted Γ at CMC = {Gamma_CMC:.4f} nm⁻²**  (experimental: 2.22 nm⁻²)")
        st.markdown(f"**Predicted ST at CMC = {ST_CMC:.2f} mN/m**  (experimental: 32.1 mN/m)")
        st.info(
            "The small gap vs experiment is expected: this code uses the simulated "
            "ΔG_ads = −44.05 kJ/mol (GAFF/GAFF-LIPID + SPC/E). "
            "The experimental ΔG_ads ≈ −46.7 kJ/mol would push Γ and ST closer to the "
            "experimental values."
        )

        G_plot = np.linspace(1e-4, Gamma_CMC, 500)
        X_plot = np.exp(eq6_lnX(G_plot, r, B))
        ST_plot = eq5_ST(G_plot, r, B)

        st.subheader("Adsorption Isotherm (Fig. 9)")
        fig1, ax1 = plt.subplots()
        ax1.plot(X_plot, G_plot, 'b-', label='Predicted isotherm')
        ax1.scatter([X_CMC], [Gamma_CMC], color='red', zorder=5,
                    label=f'Γ = {Gamma_CMC:.3f} nm⁻² at CMC')
        ax1.scatter([X_CMC], [2.22], color='green', marker='^', zorder=5, s=80,
                    label='Experimental: Γ = 2.22 nm⁻²')
        ax1.axvline(X_CMC, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xscale('log')
        ax1.set_xlabel("Mole fraction X")
        ax1.set_ylabel("Γ (nm⁻²)")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("Surface Tension vs Mole Fraction (Fig. 10)")
        fig2, ax2 = plt.subplots()
        ax2.plot(X_plot, ST_plot, 'b-', label='Predicted ST')
        ax2.scatter([X_CMC], [ST_CMC], color='red', zorder=5,
                    label=f'ST = {ST_CMC:.2f} mN/m at CMC')
        ax2.scatter([X_CMC], [32.1], color='green', marker='^', zorder=5, s=80,
                    label='Experimental: ST = 32.1 mN/m')
        ax2.axvline(X_CMC, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel("Mole fraction X")
        ax2.set_ylabel("Surface tension (mN/m)")
        ax2.legend()
        st.pyplot(fig2)