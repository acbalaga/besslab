# BESSLab — PV-only, AC-coupled (Pre-Feasibility BESS Sizing Tool)

**Purpose.**  
This Streamlit web app explores battery-energy-storage sizing and dispatch strategies for PV-only charging, AC-coupled systems at the pre-feasibility stage.  
It grew out of work done at **Emerging Power Inc. (EPI)** as part of broader renewable-energy studies, and is shared publicly to encourage learning and further improvement.

## Features
- Reads a PV 8760 profile (`hour_index, pv_mw`) and a DoD-based cycle table.  
- Simulates yearly degradation (calendar × cycle) and optional augmentation.  
- Checks if a given **contracted MW × duration** can be met under PV-only charging.  
- Computes KPIs such as compliance, PV capture, and discharge capacity factor.  
- Includes a **physics-bounded Design Advisor** that suggests feasible parameter changes.

## Getting Started
```bash
pip install -r requirements.txt
streamlit run app.py
