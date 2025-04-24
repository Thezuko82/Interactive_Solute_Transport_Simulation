import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Solute Transport Simulation", layout="wide")

st.title("Interactive Solute Transport Simulation")

# --- User Inputs ---
L = st.sidebar.slider("Domain Length (m)", 0.5, 5.0, 1.0, 0.1)
nx = st.sidebar.slider("Number of Spatial Steps", 50, 300, 100, 10)
T = st.sidebar.slider("Total Simulation Time (days)", 0.5, 10.0, 1.0, 0.1)
dt = st.sidebar.number_input("Time Step (days)", 0.0001, 0.01, 0.001, 0.0001)
v = st.sidebar.number_input("Seepage Velocity (m/day)", 0.01, 1.0, 0.1, 0.01)
D = st.sidebar.number_input("Dispersion Coefficient (m²/day)", 0.0001, 0.01, 0.001, 0.0001)
lambda_ = st.sidebar.number_input("Decay Constant (1/day)", 0.0, 1.0, 0.01, 0.01)
Kd = st.sidebar.number_input("Distribution Coefficient Kd (L/kg)", 0.0, 2.0, 0.5, 0.1)
rho_b = st.sidebar.number_input("Bulk Density (kg/L)", 1.0, 2.5, 1.6, 0.1)
theta = st.sidebar.number_input("Porosity", 0.1, 1.0, 0.4, 0.05)

# --- Derived Parameters ---
dx = L / (nx - 1)
nt = int(T / dt)
x = np.linspace(0, L, nx)
R = 1 + (rho_b * Kd) / theta

# --- Simulation Functions ---
def simulate_ADE():
    C = np.zeros(nx)
    C[0] = 100
    for _ in range(nt):
        C_new = C.copy()
        for i in range(1, nx-1):
            C_new[i] = C[i] - v * dt / (2*dx) * (C[i+1] - C[i-1]) \
                           + D * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1])
        C = C_new
    return C

def simulate_sorption_decay():
    C = np.zeros(nx)
    C[0] = 100
    for _ in range(nt):
        C_new = C.copy()
        for i in range(1, nx-1):
            C_new[i] = C[i] - (v/R) * dt / (2*dx) * (C[i+1] - C[i-1]) \
                           + (D/R) * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1]) \
                           - lambda_ * C[i] * dt / R
        C = C_new
    return C

def simulate_with_barrier():
    C = np.zeros(nx)
    C[0] = 100
    D_mod = D * np.ones(nx)
    barrier_start = int(0.45 * nx)
    barrier_end = int(0.55 * nx)
    D_mod[barrier_start:barrier_end] = D * 0.1  # reduce D in barrier zone
    for _ in range(nt):
        C_new = C.copy()
        for i in range(1, nx-1):
            C_new[i] = C[i] - (v/R) * dt / (2*dx) * (C[i+1] - C[i-1]) \
                           + (D_mod[i]/R) * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1]) \
                           - lambda_ * C[i] * dt / R
        C = C_new
    return C

# --- Run Simulations ---
C1 = simulate_ADE()
C2 = simulate_sorption_decay()
C3 = simulate_with_barrier()

# --- Plot Results ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, C1, label="ADE Only", color="blue")
ax.plot(x, C2, label="ADE + Sorption + Decay", color="red")
ax.plot(x, C3, label="ADE + Sorption + Decay + Barrier", color="green")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Concentration (mg/L)")
ax.set_title("Comparison of Solute Transport Models")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Export to CSV ---
data = pd.DataFrame({
    "Distance (m)": x,
    "ADE Only": C1,
    "With Sorption + Decay": C2,
    "With Barrier": C3
})

csv = data.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "solute_transport_results.csv", "text/csv")

