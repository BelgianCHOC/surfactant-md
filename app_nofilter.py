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


st.title("C12E6 — No Γ_max Filter (All 11 Data Points)")
st.warning(
    "This fits Equation 5 to all 11 data points including the overpacked rows "
    "Γ = 2.50 and 2.78 nm⁻². This is non-physical but shows what the unconstrained "
    "fit predicts."
)
run = st.sidebar.button("Run")

# --fit r and B based on equation 5--------

if run:
    # Fit Equation 5 to ALL 11 data points — no filter
    try:
        popt, _ = curve_fit(
            eq5_ST, Gamma_all, ST_all,
            p0=[0.3, -0.1],
            bounds=([1e-6, -np.inf], [np.inf, 0.0])
        )
        r, B = popt
    except RuntimeError:
        st.error("Curve fit failed.")
        st.stop()

# --show the results on the website--------

    st.markdown(f"**Fitted:** r = {r:.4f} nm,  B = {B:.4f} nm²")
    st.markdown("*(Compare: paper reports r = 0.279 nm, B = 0 nm² when fitting only up to Γ = 2.22)*")

# --the fitting for equation 5--------

    # Show how well eq5 fits ALL data including overpacked points
    st.subheader("Equation 5 fit vs all Table S4 data")
    fig0, ax0 = plt.subplots()
    ax0.scatter(Gamma_all, ST_all, color='black', zorder=5, label='Table S4 data (all 11 pts)')
    G_curve = np.linspace(0.01, 2.78, 400)
    ax0.plot(G_curve, eq5_ST(G_curve, r, B), 'b-',
             label=f'Eq.5 fit  r={r:.3f} nm, B={B:.3f} nm²')
    ax0.axvline(2.22, color='orange', linestyle=':', alpha=0.7, label='Exp. Γ_max = 2.22 nm⁻²')
    ax0.set_xlabel("Γ (nm⁻²)")
    ax0.set_ylabel("Surface tension (mN/m)")
    ax0.set_title("Eq. 5 fit quality (no filter)")
    ax0.legend()
    st.pyplot(fig0)

# --get the value of lnX--------

    # Solve Equation 6 for Γ at X = CMC
    lnX_CMC = np.log(X_CMC)

# --some scan that I don't understand--------

    # Scan first to check that a root exists in [1e-6, 2.78]
    G_scan = np.linspace(1e-4, 2.78, 2000)
    f_scan = eq6_lnX(G_scan, r, B) - lnX_CMC
    sign_changes = np.where(np.diff(np.sign(f_scan)))[0]

    if len(sign_changes) == 0:
        st.error(
            "Equation 6 has no root in [0, 2.78] with the unconstrained fit parameters. "
            "This can happen when a very negative B makes the isotherm non-monotonic."
        )
        st.stop()

# --single value of maximum packing at CMC, which we just want to know!!!--------

    try:
        Gamma_CMC = brentq(
            lambda G: eq6_lnX(G, r, B) - lnX_CMC,
            float(G_scan[sign_changes[0]]),
            float(G_scan[sign_changes[0] + 1]),
            xtol=1e-8
        )
    except ValueError:
        st.error("Root finding failed despite sign change detected.")
        st.stop()

# --single value of ST at CMC according to equation 5--------

    ST_CMC = eq5_ST(Gamma_CMC, r, B)

# --show the results on the website--------

    st.markdown(f"**Predicted Γ at CMC = {Gamma_CMC:.4f} nm⁻²**  (experimental: 2.22 nm⁻²)")
    st.markdown(f"**Predicted ST at CMC = {ST_CMC:.2f} mN/m**  (experimental: 32.1 mN/m)")

# --data points in the graph--------

    # Adsorption isotherm (Fig. 9)
    G_plot = np.linspace(1e-4, Gamma_CMC, 500)
    X_plot = np.exp(eq6_lnX(G_plot, r, B))
    ST_plot = eq5_ST(G_plot, r, B)

# --plot the graph--------

    st.subheader("Adsorption Isotherm (Fig. 9)")
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

# --plot the graph--------

    st.subheader("Surface Tension vs Mole Fraction (Fig. 10)")
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