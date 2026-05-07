import io
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PyMicroNanoFlow",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS PREMIUM UI
# =========================================================

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg,#f7fbff 0%,#eef4ff 45%,#ffffff 100%);
}
.block-container {
    padding-top: 1rem;
}
.hero {
    padding: 30px;
    border-radius: 28px;
    background: linear-gradient(120deg,#12347f,#2458d3,#5a9bff);
    color: white;
    box-shadow: 0 20px 45px rgba(18,52,127,0.25);
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 2.25rem;
    margin-bottom: 0.3rem;
}
.hero p {
    color: rgba(255,255,255,0.88);
    font-size: 1.03rem;
}
.badge {
    display:inline-block;
    padding:7px 13px;
    border-radius:999px;
    margin:6px 6px 0 0;
    background:rgba(255,255,255,0.16);
    border:1px solid rgba(255,255,255,0.25);
    font-size:0.82rem;
}
.card {
    padding:18px;
    border-radius:22px;
    background:rgba(255,255,255,0.88);
    border:1px solid rgba(18,52,127,0.1);
    box-shadow:0 12px 32px rgba(16,24,40,0.07);
}
.metric-label {
    color:#667085;
    font-size:0.82rem;
}
.metric-value {
    color:#101828;
    font-size:1.65rem;
    font-weight:800;
}
.note {
    padding:18px;
    border-radius:20px;
    background:#ffffff;
    border-left:6px solid #2458d3;
    box-shadow:0 8px 24px rgba(16,24,40,0.06);
}
.stTabs [data-baseweb="tab-list"] {
    gap:8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:999px;
    padding:10px 18px;
    background:rgba(36,88,211,0.08);
}
.stTabs [aria-selected="true"] {
    background:#173B8F !important;
    color:white !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class ModelParams:
    K: float
    beta: float
    kappa: float
    S: float
    lam: float
    n: float
    eta_max: float
    points: int
    tol: float
    max_nodes: int = 20000


@dataclass
class ThermalParams:
    Pr: float
    Ec: float
    Rd: float
    Q: float
    Br: float
    omega: float


@dataclass
class HybridParams:
    base_fluid: str
    particle_1: str
    particle_2: str
    phi1: float
    phi2: float


# =========================================================
# MATERIAL DATABASE
# =========================================================

MATERIALS = {
    "Water": {"rho": 997.1, "cp": 4179.0, "k": 0.613, "mu": 0.001003},
    "Ethylene Glycol": {"rho": 1115.0, "cp": 2430.0, "k": 0.253, "mu": 0.0161},
    "Engine Oil": {"rho": 884.0, "cp": 1910.0, "k": 0.144, "mu": 0.486},
    "Al2O3": {"rho": 3970.0, "cp": 765.0, "k": 40.0},
    "Cu": {"rho": 8933.0, "cp": 385.0, "k": 401.0},
    "TiO2": {"rho": 4250.0, "cp": 686.2, "k": 8.9538},
    "Graphene": {"rho": 2250.0, "cp": 2100.0, "k": 5000.0},
    "CNT": {"rho": 1600.0, "cp": 796.0, "k": 3000.0},
}


def maxwell(kf, ks, phi):
    return kf * ((ks + 2*kf - 2*phi*(kf-ks)) /
                 (ks + 2*kf + phi*(kf-ks)))


def hybrid_properties(hp: HybridParams):
    bf = MATERIALS[hp.base_fluid]
    p1 = MATERIALS[hp.particle_1]
    p2 = MATERIALS[hp.particle_2]

    phi = hp.phi1 + hp.phi2

    rho = (1 - phi)*bf["rho"] + hp.phi1*p1["rho"] + hp.phi2*p2["rho"]
    rhocp = (1 - phi)*bf["rho"]*bf["cp"] + hp.phi1*p1["rho"]*p1["cp"] + hp.phi2*p2["rho"]*p2["cp"]
    cp = rhocp / rho

    k1 = maxwell(bf["k"], p1["k"], hp.phi1)
    khnf = maxwell(k1, p2["k"], hp.phi2)

    mu = bf["mu"] / max((1 - phi), 1e-8)**2.5

    return {
        "rho_hnf": rho,
        "cp_hnf": cp,
        "k_hnf": khnf,
        "mu_hnf": mu,
        "phi_total": phi,
        "k_ratio": khnf / bf["k"],
        "mu_ratio": mu / bf["mu"],
        "rho_ratio": rho / bf["rho"],
    }


# =========================================================
# SOLVER
# =========================================================

def initial_guess(eta, p: ModelParams, branch="first"):
    if branch == "first":
        decay = 1.0
        amp = 1.0
    else:
        decay = 0.35
        amp = -2.4

    e = np.exp(-decay * eta)

    f = p.S + amp * p.lam * (1 - e)
    fp = amp * p.lam * decay * e
    fpp = -amp * p.lam * decay**2 * e
    fppp = amp * p.lam * decay**3 * e

    g = -p.n * fpp * np.exp(-0.4 * eta)
    gp = np.gradient(g, eta)

    return np.vstack([f, fp, fpp, fppp, g, gp])


def ode(eta, y, p: ModelParams):
    K = p.K
    beta = p.beta
    k = p.kappa

    A = np.maximum(eta + k, 1e-8)

    f = y[0]
    fp = y[1]
    fpp = y[2]
    fppp = y[3]
    g = y[4]
    gp = y[5]

    B = 1 + K / 2

    gpp = -(
        B * gp / A
        + (k / A) * f * gp
        - (k / A) * fp * g
        - K * (2*g + fpp + fp/A)
        - (beta/2) * (eta*gp + 3*g)
    ) / B

    rest = (
        (1 + K) * (2*fppp/A - fpp/A**2 + fp/A**3)
        - (k/A) * (fp*fpp - f*fppp)
        - (k/A**2) * (fp**2 - f*fpp)
        - (k/A**3) * f*fp
        - K * (gpp + gp/A)
        - (beta/A) * (fp + eta*fpp/2)
        - (beta/2) * (3*fpp + eta*fppp)
    )

    f4 = -rest / (1 + K)

    return np.vstack([fp, fpp, fppp, f4, gp, gpp])


def bc(ya, yb, p: ModelParams):
    return np.array([
        ya[0] - p.S,
        ya[1] - p.lam,
        ya[4] + p.n * ya[2],
        yb[1],
        yb[2],
        yb[4],
    ])


@st.cache_data(show_spinner=False)
def solve_branch(param_json, branch):
    p = ModelParams(**json.loads(param_json))

    eta = np.linspace(0, p.eta_max, p.points)
    y0 = initial_guess(eta, p, branch)

    sol = solve_bvp(
        lambda x, y: ode(x, y, p),
        lambda ya, yb: bc(ya, yb, p),
        eta,
        y0,
        tol=p.tol,
        max_nodes=p.max_nodes,
    )

    eta_dense = np.linspace(0, p.eta_max, 600)
    y = sol.sol(eta_dense)

    Cf = (1 + p.K) * (y[2, 0] + p.lam / p.kappa)
    Cm = (1 + p.K/2) * (y[5, 0] - p.n * y[2, 0] / p.kappa)

    return {
        "success": bool(sol.success),
        "message": sol.message,
        "eta": eta_dense,
        "y": y,
        "Cf": float(Cf),
        "Cm": float(Cm),
        "branch": branch,
        "nodes": int(sol.x.size),
        "niter": int(sol.niter),
    }


def heat_entropy(eta, y, tp: ThermalParams):
    fp = y[1]
    fpp = y[2]

    decay = np.sqrt(max(tp.Pr / (1 + tp.Rd + 1e-8), 0.05))
    theta = np.exp(-decay * eta)
    theta += 0.05 * tp.Ec * np.abs(fpp) / (1 + np.max(np.abs(fpp)))
    theta += 0.03 * tp.Q * np.exp(-0.5 * eta)
    theta = theta / max(theta[0], 1e-8)

    theta_p = np.gradient(theta, eta)
    Ns = (1 + tp.Rd) * theta_p**2 + tp.Br * fpp**2 + tp.omega * fp**2
    Be = ((1 + tp.Rd) * theta_p**2) / (Ns + 1e-12)
    Nu = -(1 + tp.Rd) * theta_p[0]

    return theta, theta_p, Ns, Be, Nu


def plot_line(x, ys, labels, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for y, label in zip(ys, labels):
        ax.plot(x, y, linewidth=2.5, label=label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def sweep_dual(base_params, sweep_name, values):
    rows = []

    for val in values:
        d = asdict(base_params)
        d[sweep_name] = float(val)
        p_new = ModelParams(**d)
        j = json.dumps(asdict(p_new), sort_keys=True)

        for branch in ["first", "second"]:
            try:
                r = solve_branch(j, branch)
                rows.append({
                    "parameter": float(val),
                    "branch": branch,
                    "skin_friction": r["Cf"],
                    "couple_stress": r["Cm"],
                    "success": r["success"],
                    "message": r["message"],
                })
            except Exception as e:
                rows.append({
                    "parameter": float(val),
                    "branch": branch,
                    "skin_friction": np.nan,
                    "couple_stress": np.nan,
                    "success": False,
                    "message": str(e),
                })

    return pd.DataFrame(rows)


def plot_sweep(df, ycol, title, xlabel):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for branch, marker in [("first", "o"), ("second", "s")]:
        d = df[(df["branch"] == branch) & (df["success"] == True)]
        if len(d) > 0:
            ax.plot(
                d["parameter"],
                d[ycol],
                marker=marker,
                linewidth=2.5,
                label="First solution" if branch == "first" else "Second solution"
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ycol.replace("_", " ").title())
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# =========================================================
# SIDEBAR INPUT
# =========================================================

with st.sidebar:
    st.markdown("## ⚙️ Input Parameters")

    with st.expander("Flow Model", expanded=True):
        K = st.slider("Micropolar parameter, K", 0.0, 3.0, 1.0, 0.05)
        beta = st.slider("Unsteadiness parameter, β", -6.0, 1.0, -2.0, 0.1)
        kappa = st.slider("Curvature parameter, k", 2.0, 150.0, 50.0, 1.0)
        S = st.slider("Suction/injection parameter, S", -1.0, 5.0, 2.0, 0.05)
        lam = st.slider("Stretching/shrinking parameter, λ", -1.2, 1.2, -0.1, 0.01)
        n = st.selectbox(
            "Microelement concentration, n",
            [0.0, 0.5, 1.0],
            index=1,
            format_func=lambda x: "0.0 Strong" if x == 0 else ("0.5 Weak" if x == 0.5 else "1.0 Turbulent setting")
        )

    with st.expander("Heat / Entropy", expanded=True):
        Pr = st.slider("Prandtl number, Pr", 0.7, 15.0, 6.2, 0.1)
        Ec = st.slider("Eckert number, Ec", 0.0, 1.0, 0.05, 0.01)
        Rd = st.slider("Radiation parameter, Rd", 0.0, 3.0, 0.2, 0.05)
        Q = st.slider("Heat source/sink, Q", -1.0, 1.0, 0.1, 0.05)
        Br = st.slider("Brinkman number, Br", 0.0, 1.0, 0.15, 0.01)
        omega = st.slider("Joule/magnetic entropy weight, Ω", 0.0, 1.0, 0.1, 0.01)

    with st.expander("Hybrid Nanofluid", expanded=False):
        base = st.selectbox("Base fluid", ["Water", "Ethylene Glycol", "Engine Oil"])
        particle_1 = st.selectbox("Nanoparticle 1", ["Al2O3", "Cu", "TiO2", "Graphene", "CNT"])
        particle_2 = st.selectbox("Nanoparticle 2", ["Cu", "Al2O3", "TiO2", "Graphene", "CNT"])
        phi1 = st.slider("φ1", 0.0, 0.10, 0.02, 0.005)
        phi2 = st.slider("φ2", 0.0, 0.10, 0.02, 0.005)

    with st.expander("Numerical Settings", expanded=False):
        eta_max = st.slider("η∞", 4.0, 18.0, 8.0, 0.5)
        points = st.slider("Mesh points", 80, 700, 260, 20)
        tol = st.select_slider("Tolerance", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], value=1e-4)


p = ModelParams(K, beta, kappa, S, lam, n, eta_max, points, tol)
tp = ThermalParams(Pr, Ec, Rd, Q, Br, omega)
hp = HybridParams(base, particle_1, particle_2, phi1, phi2)

param_json = json.dumps(asdict(p), sort_keys=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
<h1>PyMicroNanoFlow</h1>
<p>
Premium computational framework for unsteady micropolar hybrid nanofluid flow,
heat transport, entropy generation, dual solutions and stability analysis over curved surfaces.
</p>
<span class="badge">Micropolar fluid</span>
<span class="badge">Hybrid nanofluid</span>
<span class="badge">Curved surface</span>
<span class="badge">Dual solutions</span>
<span class="badge">Stability analysis</span>
<span class="badge">SoftwareX-ready</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MAIN COMPUTATION
# =========================================================

with st.spinner("Solving first and second solution branches..."):
    first = solve_branch(param_json, "first")
    second = solve_branch(param_json, "second")

eta = first["eta"]
y1 = first["y"]
y2 = second["y"]

theta, theta_p, Ns, Be, Nu = heat_entropy(eta, y1, tp)
props = hybrid_properties(hp)


# =========================================================
# METRICS
# =========================================================

cols = st.columns(6)
metric_data = [
    ("First Cf", first["Cf"]),
    ("Second Cf", second["Cf"]),
    ("First Cm", first["Cm"]),
    ("Second Cm", second["Cm"]),
    ("Nusselt", Nu),
    ("k_hnf/k_f", props["k_ratio"]),
]

for c, (lab, val) in zip(cols, metric_data):
    with c:
        st.markdown(f"""
        <div class="card">
        <div class="metric-label">{lab}</div>
        <div class="metric-value">{val:.5f}</div>
        </div>
        """, unsafe_allow_html=True)


if first["success"]:
    st.success(f"First solution converged | Iterations: {first['niter']} | Nodes: {first['nodes']}")
else:
    st.warning(f"First solution issue: {first['message']}")

if second["success"]:
    st.info(f"Second solution branch computed | Iterations: {second['niter']} | Nodes: {second['nodes']}")
else:
    st.warning(f"Second solution issue: {second['message']}")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Profiles",
    "📊 Skin & Couple Stress",
    "🔀 Dual Solutions",
    "🔥 Heat & Entropy",
    "🧭 Stability",
    "🧪 Hybrid Nanofluid",
    "🏗️ Architecture",
    "⬇️ Export"
])


# =========================================================
# TAB 1: PROFILES
# =========================================================

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = plot_line(
            eta,
            [y1[1], y2[1]],
            ["First solution", "Second solution"],
            "Velocity Profile",
            r"$f'(\eta)$"
        )
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig = plot_line(
            eta,
            [y1[4], y2[4]],
            ["First solution", "Second solution"],
            "Microrotation Profile",
            r"$g(\eta)$"
        )
        st.pyplot(fig, use_container_width=True)

    st.markdown("""
    <div class="note">
    <b>Meaning:</b> First and second branches are generated using different initial guesses.
    In shrinking surface problems, the second solution commonly appears near the critical region.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 2: SKIN FRICTION AND COUPLE STRESS
# =========================================================

with tab2:
    st.subheader("Skin Friction and Couple Stress Graph")

    c1, c2, c3 = st.columns(3)

    with c1:
        sweep_choice = st.selectbox(
            "Sweep parameter",
            ["lam", "K", "S", "beta", "kappa"],
            index=0,
            format_func=lambda x: {
                "lam": "λ: stretching/shrinking",
                "K": "K: micropolar",
                "S": "S: suction/injection",
                "beta": "β: unsteadiness",
                "kappa": "k: curvature",
            }[x]
        )

    with c2:
        sweep_min = st.number_input("Minimum", value=-0.9, step=0.1)

    with c3:
        sweep_max = st.number_input("Maximum", value=0.6, step=0.1)

    sweep_points = st.slider("Sweep points", 8, 50, 20)

    xlabel_map = {
        "lam": r"Stretching/shrinking parameter, $\lambda$",
        "K": r"Micropolar parameter, $K$",
        "S": r"Suction/injection parameter, $S$",
        "beta": r"Unsteadiness parameter, $\beta$",
        "kappa": r"Curvature parameter, $k$",
    }

    vals = np.linspace(sweep_min, sweep_max, sweep_points)

    with st.spinner("Running sweep for first and second branches..."):
        sweep_df = sweep_dual(p, sweep_choice, vals)

    c1, c2 = st.columns(2)

    with c1:
        fig = plot_sweep(
            sweep_df,
            "skin_friction",
            "Skin Friction Coefficient",
            xlabel_map[sweep_choice]
        )
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig = plot_sweep(
            sweep_df,
            "couple_stress",
            "Couple Stress Coefficient",
            xlabel_map[sweep_choice]
        )
        st.pyplot(fig, use_container_width=True)

    st.dataframe(sweep_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 3: DUAL SOLUTIONS
# =========================================================

with tab3:
    st.subheader("First and Second Solution Branches")

    st.markdown("""
    <div class="note">
    This section explicitly displays the two branches obtained from the nonlinear boundary-value problem.
    The first solution is generally stable and physically realizable, while the second solution is generally unstable.
    </div>
    """, unsafe_allow_html=True)

    branch_df = pd.DataFrame({
        "Branch": ["First solution", "Second solution"],
        "Solver success": [first["success"], second["success"]],
        "Skin friction": [first["Cf"], second["Cf"]],
        "Couple stress": [first["Cm"], second["Cm"]],
        "Interpretation": [
            "Usually stable / physically realizable",
            "Usually unstable / non-physically realizable"
        ]
    })

    st.dataframe(branch_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = plot_line(
            eta,
            [y1[2], y2[2]],
            ["First solution", "Second solution"],
            "Shear Profile",
            r"$f''(\eta)$"
        )
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig = plot_line(
            eta,
            [y1[5], y2[5]],
            ["First solution", "Second solution"],
            "Microrotation Gradient",
            r"$g'(\eta)$"
        )
        st.pyplot(fig, use_container_width=True)


# =========================================================
# TAB 4: HEAT AND ENTROPY
# =========================================================

with tab4:
    c1, c2 = st.columns(2)

    with c1:
        fig = plot_line(
            eta,
            [theta],
            ["Temperature"],
            "Temperature Profile",
            r"$\theta(\eta)$"
        )
        st.pyplot(fig, use_container_width=True)

        fig = plot_line(
            eta,
            [Be],
            ["Bejan number"],
            "Bejan Number",
            r"$Be$"
        )
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig = plot_line(
            eta,
            [Ns],
            ["Entropy generation"],
            "Entropy Generation Profile",
            r"$N_s$"
        )
        st.pyplot(fig, use_container_width=True)

        entropy_df = pd.DataFrame({
            "Quantity": [
                "Maximum entropy generation",
                "Average entropy generation",
                "Minimum Bejan number",
                "Average Bejan number",
                "Nusselt number proxy",
            ],
            "Value": [
                np.max(Ns),
                np.mean(Ns),
                np.min(Be),
                np.mean(Be),
                Nu,
            ]
        })

        st.dataframe(entropy_df, use_container_width=True, hide_index=True)

    st.warning(
        "Note: This prototype uses a semi-coupled thermal module. For final journal submission, replace it with the exact transformed energy equation."
    )


# =========================================================
# TAB 5: STABILITY
# =========================================================

with tab5:
    st.subheader("Stability Analysis Framework")

    st.markdown("""
    The stability analysis is based on introducing time-dependent perturbations into the steady similarity solution.
    For dual solutions, the first branch is usually stable, while the second branch is usually unstable.
    """)

    st.latex(r"f(\eta,\tau)=f_0(\eta)+e^{-\gamma \tau}F(\eta)")
    st.latex(r"g(\eta,\tau)=g_0(\eta)+e^{-\gamma \tau}G(\eta)")
    st.latex(r"\theta(\eta,\tau)=\theta_0(\eta)+e^{-\gamma \tau}H(\eta)")

    stability_df = pd.DataFrame({
        "Solution branch": ["First solution", "Second solution"],
        "Expected eigenvalue behaviour": [r"γ > 0", r"γ < 0"],
        "Disturbance behaviour": ["Decays with time", "Grows with time"],
        "Stability status": ["Stable", "Unstable"],
        "Physical meaning": ["Physically realizable", "Non-physically realizable"],
    })

    st.dataframe(stability_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="note">
    <b>Journal note:</b> For the final paper, the exact linearized eigenvalue equations must be derived from
    the proposed momentum, microrotation and energy equations. This tab prepares the computational structure
    for that rigorous stability analysis.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 6: HYBRID NANOFLUID
# =========================================================

with tab6:
    st.subheader("Hybrid Nanofluid Thermophysical Properties")

    prop_df = pd.DataFrame({
        "Property": [
            "Total volume fraction",
            "Density ratio",
            "Thermal conductivity ratio",
            "Dynamic viscosity ratio",
            "Effective density",
            "Effective heat capacity",
            "Effective thermal conductivity",
            "Effective dynamic viscosity",
        ],
        "Value": [
            props["phi_total"],
            props["rho_ratio"],
            props["k_ratio"],
            props["mu_ratio"],
            props["rho_hnf"],
            props["cp_hnf"],
            props["k_hnf"],
            props["mu_hnf"],
        ]
    })

    st.dataframe(prop_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="note">
    Selected hybrid nanofluid: <b>{base} + {particle_1} + {particle_2}</b>.
    The model uses sequential Maxwell thermal conductivity and Brinkman viscosity formulation.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 7: ARCHITECTURE
# =========================================================

with tab7:
    st.subheader("Software Architecture")

    st.code("""
PyMicroNanoFlow
│
├── Input Layer
│   ├── Flow parameters
│   ├── Heat and entropy parameters
│   ├── Hybrid nanofluid properties
│   └── Numerical settings
│
├── Solver Layer
│   ├── Nonlinear BVP solver
│   ├── First solution branch
│   ├── Second solution branch
│   └── Parametric sweep engine
│
├── Analysis Layer
│   ├── Skin friction
│   ├── Couple stress
│   ├── Temperature profile
│   ├── Entropy generation
│   ├── Bejan number
│   └── Stability framework
│
└── Export Layer
    ├── CSV output
    ├── Parameter JSON
    └── Publication-ready graphs
    """)

    st.markdown("""
    <div class="note">
    <b>SoftwareX positioning:</b> The contribution is an interactive, reproducible computational framework
    transforming static micropolar curved-surface studies into reusable scientific software.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TAB 8: EXPORT
# =========================================================

with tab8:
    st.subheader("Export Results")

    result_df = pd.DataFrame({
        "eta": eta,
        "f_first": y1[0],
        "velocity_first": y1[1],
        "shear_first": y1[2],
        "microrotation_first": y1[4],
        "f_second": y2[0],
        "velocity_second": y2[1],
        "shear_second": y2[2],
        "microrotation_second": y2[4],
        "temperature": theta,
        "entropy_generation": Ns,
        "bejan_number": Be,
    })

    st.dataframe(result_df.head(80), use_container_width=True, hide_index=True)

    st.download_button(
        "Download full results CSV",
        data=result_df.to_csv(index=False),
        file_name="pymicronanoflow_full_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    export_params = {
        "model_parameters": asdict(p),
        "thermal_parameters": asdict(tp),
        "hybrid_parameters": asdict(hp),
        "hybrid_properties": props,
        "first_solution": {
            "success": first["success"],
            "skin_friction": first["Cf"],
            "couple_stress": first["Cm"],
        },
        "second_solution": {
            "success": second["success"],
            "skin_friction": second["Cf"],
            "couple_stress": second["Cm"],
        }
    }

    st.download_button(
        "Download parameter JSON",
        data=json.dumps(export_params, indent=4),
        file_name="pymicronanoflow_parameters.json",
        mime="application/json",
        use_container_width=True
    )


st.markdown("---")
st.caption(
    "PyMicroNanoFlow premium prototype | For journal submission, replace the heat/entropy surrogate with the exact transformed energy and entropy equations."
)
