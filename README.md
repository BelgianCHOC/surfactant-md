# C12E6 Surfactant Adsorption Isotherm — MD-MTT Framework

An interactive Streamlit app that implements the molecular dynamics–molecular thermodynamics theory (MD-MTT) framework from Li, Amador & Wilson (2024) to predict the surface tension isotherm and surface excess concentration of the non-ionic surfactant **C12E6** at the water–vacuum interface.

---

## Scientific background

The app reproduces the workflow described in **Section 3.7** of:

> Jing Li, Carlos Amador, Mark R. Wilson.  
> *Computational predictions of interfacial tension, surface tension, and surfactant adsorption isotherms.*  
> Phys. Chem. Chem. Phys., 2024, **26**, 12107–12120.  
> DOI: [10.1039/D3CP06170A](https://doi.org/10.1039/D3CP06170A)

### Equation 5 — surface equation of state

$$\frac{\gamma_0 - \gamma}{k_BT} = \frac{\Gamma}{(1-Z)^2} + B\Gamma^2$$

where $Z = \Gamma \pi r^2$ is the packing fraction, $r$ is the surfactant radius (nm), and $B$ is the second virial coefficient (nm²).  
This equation is **fitted** to the simulated surface tension data in Table S4 (C12E6 column) to obtain $r$ and $B$.

### Equation 6 — adsorption isotherm

$$\ln X = \frac{\Delta G_\text{ads}}{k_BT} + \ln\frac{Z}{1-Z} + \frac{3Z - 2Z^2}{(1-Z)^2} + 2B\Gamma$$

where $X$ is the bulk mole fraction.  
Given the fitted $r$, $B$, and $\Delta G_\text{ads} = -44.05\ \text{kJ mol}^{-1}$ (Table 9), this equation predicts $\Gamma$ at any bulk concentration.

### Self-consistent iterative algorithm

Because the data cutoff used for fitting Equation 5 depends on $\Gamma_\text{max}$, which is itself predicted by Equation 6, the two steps are iterated until convergence:

```
1. Start with an initial guess for Γ_max
2. Fit Equation 5 to Table S4 data where Γ ≤ Γ_max  →  r, B
3. Solve Equation 6 at X = X_CMC  →  new Γ_max
4. Repeat until |ΔΓ_max| < tolerance
5. Compute ST at the converged Γ_max from Equation 5
```

---

## Input data

| Source | Value used |
|--------|-----------|
| Table S4 (ESI) — C12E6 column | Simulated ST vs Γ (GAFF/GAFF-LIPID + OPC4) |
| Table 9 | $\Delta G_\text{ads} = -44.05\ \text{kJ mol}^{-1}$ (GAFF/GAFF-LIPID + SPC/E) |
| Experimental CMC | $X_\text{CMC} = 1.477 \times 10^{-6}$ mole fraction |

---

## Results

| | Predicted (this app) | Experimental |
|--|--|--|
| $\Gamma$ at CMC | ~2.10 nm⁻² | 2.22 nm⁻² |
| ST at CMC | ~34 mN m⁻¹ | 32.1 mN m⁻¹ |

The small discrepancy is expected: the app uses the **simulated** $\Delta G_\text{ads} = -44.05\ \text{kJ mol}^{-1}$, while the experimental value is $-46.7\ \text{kJ mol}^{-1}$.

---

## Installation

```bash
git clone https://github.com/BelgianCHOC/surfactant-md.git
cd surfactant-md
pip install streamlit numpy scipy matplotlib
```

## Usage

```bash
streamlit run app.py
```

Then open the sidebar to set the **initial Γ_max guess** and **convergence tolerance**, and click **Run**.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web UI |
| `numpy` | Numerical arrays |
| `scipy` | `curve_fit` (Eq. 5 fitting) and `brentq` (root-finding for Eq. 6) |
| `matplotlib` | Plots |
