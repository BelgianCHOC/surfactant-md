import streamlit as st
import numpy as np
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt

T = 298.15
kB = 1.380649e-23
R  = 8.314
kBT_eff = kB * T * 1e18 * 1e3
DGads_kBT = -44050.0 / (R * T)
gamma0 = 70.66
X_CMC  = 1.477e-6

Gamma_all = np.array([0.00, 0.28, 0.56, 0.83, 1.11,
                       1.39, 1.67, 1.94, 2.22, 2.50, 2.78])
ST_all    = np.array([70.66, 69.36, 65.59, 59.30, 54.41,
                       48.60, 44.29, 38.70, 32.18, 25.69, 17.15])

def eq5_ST(Gamma, r, B):
    Z = Gamma * np.pi * r**2
    return gamma0 - kBT_eff * (Gamma / (1.0 - Z)**2 + B * Gamma**2)

def eq6_lnX(Gamma, r, B):
    Z = Gamma * np.pi * r**2
    return (DGads_kBT
            + np.log(Z / (1.0 - Z))
            + (3*Z - 2*Z**2) / (1.0 - Z)**2
            + 2*B*Gamma)

def run_20_iterations(Gamma_max_init):
    Gamma_max = Gamma_max_init
    history = []
    r_last, B_last, Gamma_last = None, None, None

    for i in range(20):
        mask = Gamma_all <= Gamma_max + 1e-9
        G_fit = Gamma_all[mask]
        S_fit = ST_all[mask]

        if len(G_fit) < 3:
            st.warning(f"Stopped at iteration {i+1}: fewer than 3 data points below Γ_max = {Gamma_max:.3f}.")
            break

        try:
            popt, _ = curve_fit(
                eq5_ST, G_fit, S_fit,
                p0=[0.3, -0.1],
                bounds=([1e-6, -np.inf], [np.inf, 0.0])
            )
            r, B = popt
        except RuntimeError:
            st.warning(f"Curve fit failed at iteration {i+1}.")
            break

        lnX_CMC = np.log(X_CMC)
        try:
            Gamma_new = brentq(
                lambda G: eq6_lnX(G, r, B) - lnX_CMC,
                1e-6, 2.78,
                xtol=1e-8
            )
        except ValueError:
            st.warning(f"Root finding failed at iteration {i+1}.")
            break

        ST_new = eq5_ST(Gamma_new, r, B)
        r_last, B_last, Gamma_last = r, B, Gamma_new

        history.append({
            "Iteration": i + 1,
            "Γ_max (nm⁻²)": round(Gamma_max, 4),
            "Data up to (nm⁻²)": round(float(G_fit[-1]), 2),
            "r (nm)": round(r, 4),
            "B (nm²)": round(B, 4),
            "Γ_CMC (nm⁻²)": round(Gamma_new, 4),
            "ST at CMC (mN/m)": round(ST_new, 2),
        })

        Gamma_max = Gamma_new

    return history, r_last, B_last, Gamma_last


st.title("C12E6 — First 20 Iterations (No Early Stop)")
st.sidebar.header("Settings")

Gamma_init = st.sidebar.slider(
    "Initial Γ_max guess (nm⁻²)",
    min_value=0.28, max_value=2.78, value=2.78, step=0.01
)
run = st.sidebar.button("Run")

if run:
    history, r, B, Gamma_CMC = run_20_iterations(Gamma_init)

    if not history:
        st.error("Failed at the first iteration. Try a larger initial Γ_max.")
    else:
        import pandas as pd
        df = pd.DataFrame(history)

        st.subheader("Iteration table")
        st.dataframe(df)

        st.markdown(f"**Final (iteration {len(history)}):** r = {r:.4f} nm,  B = {B:.4f} nm²")
        st.markdown(f"**Final Γ at CMC = {Gamma_CMC:.4f} nm⁻²**  (experimental: 2.22 nm⁻²)")
        st.markdown(f"**Final ST at CMC = {eq5_ST(Gamma_CMC, r, B):.2f} mN/m**  (experimental: 32.1 mN/m)")

        # Convergence plots: Γ_CMC and ST per iteration
        st.subheader("Convergence plots")
        fig0, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))

        ax_l.plot(df["Iteration"], df["Γ_CMC (nm⁻²)"], 'b-o', markersize=5)
        ax_l.axhline(2.22, color='green', linestyle='--', alpha=0.7, label='Exp. 2.22 nm⁻²')
        ax_l.set_xlabel("Iteration")
        ax_l.set_ylabel("Γ_CMC (nm⁻²)")
        ax_l.set_title("Γ_CMC per iteration")
        ax_l.legend()

        ax_r.plot(df["Iteration"], df["ST at CMC (mN/m)"], 'r-o', markersize=5)
        ax_r.axhline(32.1, color='green', linestyle='--', alpha=0.7, label='Exp. 32.1 mN/m')
        ax_r.set_xlabel("Iteration")
        ax_r.set_ylabel("ST at CMC (mN/m)")
        ax_r.set_title("ST at CMC per iteration")
        ax_r.legend()

        plt.tight_layout()
        st.pyplot(fig0)

        # Final isotherm and ST vs X using last iteration's r, B
        G_plot = np.linspace(1e-4, Gamma_CMC, 500)
        X_plot = np.exp(eq6_lnX(G_plot, r, B))
        ST_plot = eq5_ST(G_plot, r, B)
        ST_CMC = eq5_ST(Gamma_CMC, r, B)

        st.subheader("Adsorption Isotherm — final iteration (Fig. 9)")
        fig1, ax1 = plt.subplots()
        ax1.plot(X_plot, G_plot, 'b-', label='Predicted isotherm')
        ax1.scatter([X_CMC], [Gamma_CMC], color='red', zorder=5,
                    label=f'Predicted: Γ = {Gamma_CMC:.3f} nm⁻²')
        ax1.scatter([X_CMC], [2.22], color='green', marker='^', zorder=5, s=80,
                    label='Experimental: Γ = 2.22 nm⁻²')
        ax1.axvline(X_CMC, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xscale('log')
        ax1.set_xlabel("Mole fraction X")
        ax1.set_ylabel("Γ (nm⁻²)")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("Surface Tension vs Mole Fraction — final iteration (Fig. 10)")
        fig2, ax2 = plt.subplots()
        ax2.plot(X_plot, ST_plot, 'b-', label='Predicted ST')
        ax2.scatter([X_CMC], [ST_CMC], color='red', zorder=5,
                    label=f'Predicted: ST = {ST_CMC:.2f} mN/m')
        ax2.scatter([X_CMC], [32.1], color='green', marker='^', zorder=5, s=80,
                    label='Experimental: ST = 32.1 mN/m')
        ax2.axvline(X_CMC, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xscale('log')
        ax2.set_xlabel("Mole fraction X")
        ax2.set_ylabel("Surface tension (mN/m)")
        ax2.legend()
        st.pyplot(fig2)