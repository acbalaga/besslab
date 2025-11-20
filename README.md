# BESSLab — PV-only, AC-coupled (Pre-Feasibility BESS Sizing Tool)

An interactive Streamlit app for exploring **battery energy storage** sizing and behavior when it charges only from **PV** and delivers through an **AC-coupled** point of interconnection.

## What you can do
- Test whether a chosen **contracted MW × duration** can be served using PV first and the BESS for any shortfall.
- See how performance changes with **round-trip efficiency, state-of-charge limits, availability, and degradation** settings.
- Try optional **augmentation** strategies to keep the system on target across the project life.
- View clear charts for **end-of-year capability**, **PV vs. BESS energy delivered**, and **typical daily profiles**.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL in your browser to launch the app.

## Using the app
1. **Upload or use defaults.** Provide a PV 8760 CSV (`hour_index, pv_mw`) and, optionally, a cycle-model Excel file. If you skip uploads, the app uses included sample data.
2. **Set your target.** Enter the contracted power (MW) and desired duration (hours).
3. **Adjust assumptions.** Use sidebar controls for efficiency, state-of-charge limits, availability, and augmentation options.
4. **Review results.** Check the compliance summary, flags, and charts to see how well the setup holds over time.

## Tips for best results
- Keep `hour_index` consecutive (0–8759 or 1–8760); the app auto-corrects the starting index.
- If you see frequent shortfalls, try widening the SOC range, improving efficiency, or enabling augmentation.
- Use the Design Advisor panel for quick, bounded suggestions when the system struggles to meet the contract.

## Feedback
This tool originated from studies at **Emerging Power Inc. (EPI)**. Feedback and improvement ideas are welcome—please reach out with suggestions or issues.
