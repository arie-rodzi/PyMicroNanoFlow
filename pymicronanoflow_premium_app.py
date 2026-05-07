# ============================================================
# PyMicroNanoFlow
# Premium Streamlit App for:
# "A Computational Framework for Entropy Generation and Stability
# Analysis of Unsteady Micropolar Hybrid Nanofluid Flow over a Curved Surface"
# ============================================================

import io
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from scipy.integrate import solve_bvp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(
    page_title="PyMicroNanoFlow | Micropolar Hybrid Nanofluid Solver",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('''
<style>
.main {background: linear-gradient(135deg, #F7FAFF 0%, #EEF4FF 45%, #FFFFFF 100%);} 
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {padding: 28px 30px; border-radius: 28px; background: linear-gradient(120deg, #173B8F 0%, #2458D3 55%, #4F8DF7 100%); color: white; box-shadow: 0 18px 45px rgba(23,59,143,0.25); margin-bottom: 18px;}
.hero h1 {font-size: 2.15rem; line-height: 1.14; margin-bottom: 0.35rem;}
.hero p {color: rgba(255,255,255,0.88); font-size: 1.02rem;}
.badge {display: inline-block; padding: 7px 12px; margin-right: 7px; margin-top: 8px; border-radius: 999px; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.24); font-size: 0.83rem;}
.metric-card {background: rgba(255,255,255,0.88); border: 1px solid rgba(23,59,143,0.10); border-radius: 24px; padding: 18px 18px; box-shadow: 0 10px 30px rgba(16,24,40,0.07);}
.metric-label {color: #667085; font-size: 0.82rem;}
.metric-value {color: #122033; font-size: 1.8rem; font-weight: 800; margin-top: 4px;}
.section-card {background: rgba(255,255,255,0.82); border: 1px solid rgba(23,59,143,0.10); border-radius: 24px; padding: 20px 22px; box-shadow: 0 10px 28px rgba(16,24,40,0.06); margin-bottom: 18px;}
.small-muted {color: #667085; font-size: 0.88rem;}
hr {border: none; height: 1px; background: linear-gradient(90deg, rgba(23,59,143,0.05), rgba(23,59,143,0.28), rgba(23,59,143,0.05));}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {border-radius: 999px; padding: 10px 18px; background: rgba(23,59,143,0.08);}
.stTabs [aria-selected="true"] {background: #173B8F !important; color: white !important;}
</style>
''', unsafe_allow_html=True)

@dataclass
class ModelParams:
    K: float = 1.0
    beta: float = -2.0
    kappa: float = 50.0
    S: float = 2.0
    lam: float = -0.1
    n: float = 0.5
    eta_max: float = 8.0
    points: int = 260
    tol: float = 1e-4
    max_nodes: int = 20000

@dataclass
class ThermalParams:
    Pr: float = 6.2
    Ec: float = 0.05
    Rd: float = 0.2
    Q: float = 0.1
    Br: float = 0.15
    omega: float = 0.1

@dataclass
class HybridFluidParams:
    base_fluid: str = "Water"
    nanoparticle_1: str = "Al2O3"
    nanoparticle_2: str = "Cu"
    phi1: float = 0.02
    phi2: float = 0.02

MATERIALS = {
    "Water": {"rho": 997.1, "cp": 4179.0, "k": 0.613, "mu": 0.001003},
    "Ethylene glycol": {"rho": 1115.0, "cp": 2430.0, "k": 0.253, "mu": 0.0161},
    "Engine oil": {"rho": 884.0, "cp": 1910.0, "k": 0.144, "mu": 0.486},
    "Al2O3": {"rho": 3970.0, "cp": 765.0, "k": 40.0},
    "Cu": {"rho": 8933.0, "cp": 385.0, "k": 401.0},
    "TiO2": {"rho": 4250.0, "cp": 686.2, "k": 8.9538},
    "Graphene": {"rho": 2250.0, "cp": 2100.0, "k": 5000.0},
    "CNT": {"rho": 1600.0, "cp": 796.0, "k": 3000.0},
}

def maxwell_k(kf: float, ks: float, phi: float) -> float:
    num = ks + 2 * kf - 2 * phi * (kf - ks)
    den = ks + 2 * kf + phi * (kf - ks)
    return kf * (num / den)

def hybrid_properties(hp: HybridFluidParams) -> dict:
    bf = MATERIALS[hp.base_fluid]
    s1 = MATERIALS[hp.nanoparticle_1]
    s2 = MATERIALS[hp.nanoparticle_2]
    phi1, phi2 = hp.phi1, hp.phi2
    phi = phi1 + phi2
    rho = (1 - phi) * bf["rho"] + phi1 * s1["rho"] + phi2 * s2["rho"]
    rhocp = (1 - phi) * bf["rho"] * bf["cp"] + phi1 * s1["rho"] * s1["cp"] + phi2 * s2["rho"] * s2["cp"]
    cp = rhocp / rho
    k_temp = maxwell_k(bf["k"], s1["k"], phi1)
    k_eff = maxwell_k(k_temp, s2["k"], phi2)
    mu = bf["mu"] / max((1 - phi), 1e-8) ** 2.5
    return {"rho_hnf": rho, "cp_hnf": cp, "k_hnf": k_eff, "mu_hnf": mu, "phi_total": phi, "rho_ratio": rho / bf["rho"], "k_ratio": k_eff / bf["k"], "mu_ratio": mu / bf["mu"]}

def initial_guess(eta: np.ndarray, p: ModelParams) -> np.ndarray:
    e = np.exp(-eta)
    f = p.S + p.lam * (1.0 - e)
    fp = p.lam * e
    fpp = -p.lam * e
    fppp = p.lam * e
    g = -p.n * fpp * e
    gp = np.gradient(g, eta)
    return np.vstack([f, fp, fpp, fppp, g, gp])

def micropolar_ode(eta: np.ndarray, y: np.ndarray, p: ModelParams) -> np.ndarray:
    K, beta, k = p.K, p.beta, p.kappa
    A = np.maximum(eta + k, 1e-7)
    f, fp, fpp, fppp, g, gp = y
    B = 1.0 + K / 2.0
    gpp = -(B * gp / A + (k / A) * f * gp - (k / A) * fp * g - K * (2 * g + fpp + fp / A) - (beta / 2.0) * (eta * gp + 3 * g)) / B
    rest = ((1 + K) * (2 * fppp / A - fpp / A**2 + fp / A**3) - (k / A) * (fp * fpp - f * fppp) - (k / A**2) * (fp**2 - f * fpp) - (k / A**3) * f * fp - K * (gpp + gp / A) - (beta / A) * (fp + (eta / 2.0) * fpp) - (beta / 2.0) * (3 * fpp + eta * fppp))
    f4 = -rest / (1 + K)
    return np.vstack([fp, fpp, fppp, f4, gp, gpp])

def bc(ya: np.ndarray, yb: np.ndarray, p: ModelParams) -> np.ndarray:
    return np.array([ya[0] - p.S, ya[1] - p.lam, ya[4] + p.n * ya[2], yb[1], yb[2], yb[4]])

@st.cache_data(show_spinner=False)
def solve_micropolar_cached(param_json: str):
    p = ModelParams(**json.loads(param_json))
    eta = np.linspace(0, p.eta_max, int(p.points))
    y0 = initial_guess(eta, p)
    if not SCIPY_AVAILABLE:
        return {"success": False, "message": "SciPy is not available. Install scipy to activate solve_bvp.", "eta": eta, "y": y0, "skin_friction": np.nan, "couple_stress": np.nan, "nodes": len(eta), "niter": 0}
    sol = solve_bvp(lambda x, y: micropolar_ode(x, y, p), lambda ya, yb: bc(ya, yb, p), eta, y0, tol=p.tol, max_nodes=p.max_nodes, verbose=0)
    eta_dense = np.linspace(0, p.eta_max, 600)
    y = sol.sol(eta_dense)
    skin_friction = (1 + p.K) * (y[2, 0] + p.lam / p.kappa)
    couple_stress = (1 + p.K / 2) * (y[5, 0] - p.n * y[2, 0] / p.kappa)
    return {"success": bool(sol.success), "message": sol.message, "eta": eta_dense, "y": y, "skin_friction": float(skin_friction), "couple_stress": float(couple_stress), "nodes": int(sol.x.size), "niter": int(sol.niter)}

def compute_heat_entropy(eta: np.ndarray, y: np.ndarray, tp: ThermalParams):
    fp, fpp = y[1], y[2]
    thermal_decay = np.sqrt(max(tp.Pr / (1 + tp.Rd + 1e-9), 0.02))
    theta = np.exp(-thermal_decay * eta) * (1 + 0.08 * tp.Q * np.exp(-0.5 * eta))
    theta = theta + 0.04 * tp.Ec * np.abs(fpp) / (1 + np.max(np.abs(fpp)) + 1e-9)
    theta = theta / max(theta[0], 1e-9)
    theta_prime = np.gradient(theta, eta)
    Ns = (1 + tp.Rd) * theta_prime**2 + tp.Br * fpp**2 + tp.omega * fp**2
    Be = ((1 + tp.Rd) * theta_prime**2) / (Ns + 1e-12)
    Nu = -theta_prime[0] * (1 + tp.Rd)
    return theta, theta_prime, Ns, Be, float(Nu)

def make_plot(x, series, labels, title, xlabel=r"$\eta$", ylabel="", height=4.2):
    fig, ax = plt.subplots(figsize=(7.8, height))
    for s, lab in zip(series, labels):
        ax.plot(x, s, linewidth=2.4, label=lab)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

def dataframe_from_solution(result, theta, Ns, Be):
    eta, y = result["eta"], result["y"]
    return pd.DataFrame({"eta": eta, "f": y[0], "f_prime_velocity": y[1], "f_double_prime": y[2], "g_microrotation": y[4], "temperature_theta": theta, "entropy_generation_Ns": Ns, "Bejan_number_Be": Be})

with st.sidebar:
    st.markdown("## ⚙️ Model Inputs")
    st.caption("Base parameters for the unsteady micropolar curved-surface model.")
    with st.expander("1. Flow and micropolar parameters", expanded=True):
        K = st.slider("Micropolar parameter, K", 0.0, 3.0, 1.0, 0.05)
        beta = st.slider("Unsteadiness parameter, β", -6.0, 1.0, -2.0, 0.1)
        kappa = st.slider("Curvature parameter, k", 2.0, 120.0, 50.0, 1.0)
        S = st.slider("Suction/injection parameter, S", -1.0, 5.0, 2.0, 0.05)
        lam = st.slider("Stretching/shrinking parameter, λ", -1.2, 1.2, -0.1, 0.01)
        n = st.selectbox("Microelement concentration, n", [0.0, 0.5, 1.0], index=1, format_func=lambda x: "0.0 Strong concentration" if x == 0 else ("0.5 Weak concentration" if x == 0.5 else "1.0 Turbulent-flow setting"))
    with st.expander("2. Heat and entropy parameters", expanded=True):
        Pr = st.slider("Prandtl number, Pr", 0.7, 15.0, 6.2, 0.1)
        Ec = st.slider("Eckert number, Ec", 0.0, 1.0, 0.05, 0.01)
        Rd = st.slider("Radiation parameter, Rd", 0.0, 3.0, 0.2, 0.05)
        Q = st.slider("Heat source/sink parameter, Q", -1.0, 1.0, 0.1, 0.05)
        Br = st.slider("Brinkman number, Br", 0.0, 1.0, 0.15, 0.01)
        omega = st.slider("Magnetic/Joule entropy weight, Ω", 0.0, 1.0, 0.1, 0.01)
    with st.expander("3. Hybrid nanofluid properties", expanded=False):
        base_fluid = st.selectbox("Base fluid", ["Water", "Ethylene glycol", "Engine oil"])
        np1 = st.selectbox("Nanoparticle 1", ["Al2O3", "Cu", "TiO2", "Graphene", "CNT"], index=0)
        np2 = st.selectbox("Nanoparticle 2", ["Cu", "Al2O3", "TiO2", "Graphene", "CNT"], index=0)
        phi1 = st.slider("Volume fraction φ1", 0.0, 0.10, 0.02, 0.005)
        phi2 = st.slider("Volume fraction φ2", 0.0, 0.10, 0.02, 0.005)
    with st.expander("4. Numerical settings", expanded=False):
        eta_max = st.slider("η∞ truncation", 4.0, 16.0, 8.0, 0.5)
        points = st.slider("Initial mesh points", 80, 600, 260, 20)
        tol = st.select_slider("Solver tolerance", options=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5], value=1e-4)
    st.button("🚀 Run Simulation", use_container_width=True)

p = ModelParams(K=K, beta=beta, kappa=kappa, S=S, lam=lam, n=n, eta_max=eta_max, points=points, tol=tol)
tp = ThermalParams(Pr=Pr, Ec=Ec, Rd=Rd, Q=Q, Br=Br, omega=omega)
hp = HybridFluidParams(base_fluid=base_fluid, nanoparticle_1=np1, nanoparticle_2=np2, phi1=phi1, phi2=phi2)

st.markdown('''
<div class="hero">
    <h1>PyMicroNanoFlow: Premium Computational Framework</h1>
    <p>Interactive solver for unsteady micropolar hybrid nanofluid flow, heat transport, entropy generation and stability-oriented analysis over a curved surface.</p>
    <span class="badge">Micropolar fluid</span><span class="badge">Hybrid nanofluid</span><span class="badge">Curved surface</span><span class="badge">Entropy generation</span><span class="badge">SoftwareX-ready prototype</span>
</div>
''', unsafe_allow_html=True)

param_json = json.dumps(asdict(p), sort_keys=True)
with st.spinner("Solving nonlinear boundary-value problem..."):
    result = solve_micropolar_cached(param_json)
eta, y = result["eta"], result["y"]
theta, theta_prime, Ns, Be, Nu = compute_heat_entropy(eta, y, tp)
props = hybrid_properties(hp)
df = dataframe_from_solution(result, theta, Ns, Be)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [("Skin friction", f"{result['skin_friction']:.5f}" if np.isfinite(result["skin_friction"]) else "N/A"), ("Couple stress", f"{result['couple_stress']:.5f}" if np.isfinite(result["couple_stress"]) else "N/A"), ("Nusselt proxy", f"{Nu:.5f}"), ("Entropy max", f"{np.max(Ns):.5f}"), ("k_hnf/k_f", f"{props['k_ratio']:.3f}")]
for col, (label, value) in zip([m1, m2, m3, m4, m5], metrics):
    with col:
        st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>''', unsafe_allow_html=True)

if result["success"]:
    st.success(f"Solver converged successfully | Iterations: {result['niter']} | Nodes: {result['nodes']}")
else:
    st.warning(f"Solver status: {result['message']}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Profiles", "🔥 Heat & Entropy", "🧪 Hybrid Nanofluid", "✅ Validation", "🏗️ Architecture", "⬇️ Export"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(make_plot(eta, [y[1]], [r"Velocity $f'(\eta)$"], "Velocity Profile", ylabel=r"$f'(\eta)$"), use_container_width=True)
    with c2:
        st.pyplot(make_plot(eta, [y[4]], [r"Microrotation $g(\eta)$"], "Microrotation Profile", ylabel=r"$g(\eta)$"), use_container_width=True)
    st.markdown('''<div class="section-card"><b>Interpretation guide.</b> The velocity and microrotation profiles are the core outputs of the base micropolar curved-surface model. For shrinking surfaces, dual-solution behaviour may appear depending on λ, β, S and k. This prototype solves one branch based on the selected initial guess; a future branch-tracking module can be added for first/second solution tracing.</div>''', unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(make_plot(eta, [theta], [r"Temperature $\theta(\eta)$"], "Heat Transport Profile", ylabel=r"$\theta(\eta)$"), use_container_width=True)
        st.pyplot(make_plot(eta, [Be], ["Bejan number"], "Bejan Number Distribution", ylabel="Be"), use_container_width=True)
    with c2:
        st.pyplot(make_plot(eta, [Ns], ["Entropy generation"], "Entropy Generation Profile", ylabel=r"$N_s$"), use_container_width=True)
        entropy_table = pd.DataFrame({"Quantity": ["Maximum entropy generation", "Average entropy generation", "Minimum Bejan number", "Average Bejan number", "Nusselt proxy"], "Value": [np.max(Ns), np.mean(Ns), np.min(Be), np.mean(Be), Nu]})
        st.dataframe(entropy_table, use_container_width=True, hide_index=True)
    st.info("For final journal submission, replace the semi-coupled thermal surrogate with the full transformed energy equation. The UI, entropy engine, export module and plotting pipeline are already prepared.")

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Hybrid Nanofluid Thermophysical Properties")
    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        st.write("Selected mixture:")
        st.markdown(f"""
- Base fluid: **{hp.base_fluid}**
- Nanoparticle 1: **{hp.nanoparticle_1}**, φ1 = **{hp.phi1:.3f}**
- Nanoparticle 2: **{hp.nanoparticle_2}**, φ2 = **{hp.phi2:.3f}**
- Total volume fraction: **{props['phi_total']:.3f}**
""")
    with pcol2:
        prop_df = pd.DataFrame({"Property": ["Density", "Specific heat", "Thermal conductivity", "Dynamic viscosity", "k ratio", "μ ratio"], "Symbol": [r"ρ_hnf", r"cp_hnf", r"k_hnf", r"μ_hnf", r"k_hnf/k_f", r"μ_hnf/μ_f"], "Value": [props["rho_hnf"], props["cp_hnf"], props["k_hnf"], props["mu_hnf"], props["k_ratio"], props["mu_ratio"]]})
        st.dataframe(prop_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Current implementation uses sequential Maxwell thermal conductivity and Brinkman viscosity. These can be replaced with any selected hybrid nanofluid model required by the target paper.")

with tab4:
    st.subheader("Validation and Benchmarking Module")
    st.markdown("""
This module is designed to reproduce benchmark cases from the base micropolar curved-surface model.

| Case | Setting | Expected use |
|---|---|---|
| Flat surface limit | k → large | Compare with classical flat stretching/shrinking solution |
| No micropolar effect | K = 0 | Reduce to viscous-fluid limit |
| Steady limit | β = 0 | Compare with steady curved-surface model |
| Stretching case | λ > 0 | Check skin-friction benchmark |
| Shrinking case | λ < 0 | Check dual-solution region |
""")
    validation_df = pd.DataFrame({"Selected parameter": ["K", "β", "k", "S", "λ", "n"], "Current value": [K, beta, kappa, S, lam, n], "Validation note": ["K=0 gives non-micropolar reference", "β=0 gives steady reference", "Large k approximates flat surface", "S>0 suction stabilizes shrinking solution", "λ>0 stretching; λ<0 shrinking", "n=0 strong; n=0.5 weak concentration"]})
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    st.warning("For SoftwareX submission, add one table that numerically matches published benchmark values. This tab is ready for that table.")

with tab5:
    st.subheader("Software Architecture")
    st.markdown("""
```text
User Input Layer
    ├── Flow parameters
    ├── Heat and entropy parameters
    ├── Hybrid nanofluid material selection
    └── Numerical solver settings

Computation Layer
    ├── Thermophysical property engine
    ├── Nonlinear BVP solver
    ├── Heat transport module
    ├── Entropy generation module
    └── Validation engine

Visualization Layer
    ├── Velocity profile
    ├── Microrotation profile
    ├── Temperature profile
    ├── Entropy profile
    └── Bejan number profile

Export Layer
    ├── CSV results
    ├── JSON parameter file
    └── Publication-ready figures
```
""")
    st.markdown('''<div class="section-card"><b>SoftwareX positioning.</b> The main contribution is not only solving the model, but turning a static mathematical fluid-flow study into a reusable, interactive and reproducible computational research platform.</div>''', unsafe_allow_html=True)

with tab6:
    st.subheader("Export Simulation Results")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("Download results as CSV", data=csv_buffer.getvalue(), file_name="pymicronanoflow_results.csv", mime="text/csv", use_container_width=True)
    params_export = {"model_parameters": asdict(p), "thermal_parameters": asdict(tp), "hybrid_fluid_parameters": asdict(hp), "hybrid_properties": props, "solver_success": result["success"], "solver_message": result["message"], "skin_friction": result["skin_friction"], "couple_stress": result["couple_stress"], "nusselt_proxy": Nu}
    st.download_button("Download parameters as JSON", data=json.dumps(params_export, indent=4), file_name="pymicronanoflow_parameters.json", mime="application/json", use_container_width=True)
    st.dataframe(df.head(60), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('''<div class="small-muted"><b>Research note:</b> This is a premium prototype. For final manuscript submission, the thermal equation and entropy equation should be written exactly according to the proposed mathematical formulation, and validation should be completed against published benchmark tables.</div>''', unsafe_allow_html=True)
