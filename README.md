# cloud-cost-intelligence-platform-for-retail
Retail cloud cost analysis with anomaly detection, budget forecasting, and optimization agents.
# Cloud Cost Intelligence for Retail

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

A lightweight, open‑source cloud cost intelligence platform designed for retail organizations. Upload your cloud billing CSV and instantly receive budget forecasts, anomaly detection, MoM driver analysis, unit economics, and idle resource identification – all powered by five autonomous analysis agents.

## ✨ Features

- **Budget Guard** – Month‑end projection vs. budget with savings needed.
- **Anomaly Investigator** – Rolling Z‑score (14‑day) detects unusual daily spend.
- **Spend Driver** – Decomposes month‑over‑month change by service and department.
- **Unit Economics** – Ranks services by cost per usage unit.
- **Optimization Scout** – Identifies high‑cost, low‑usage resources (idle candidates).

## 🖥️ Demo

![Screenshot of Overview tab](screenshot.png)  
*Replace with your actual screenshot*

## 📦 Technologies

- Python 3.10+
- Streamlit (interactive dashboard)
- Pandas (data processing)
- NumPy (synthetic data & statistics)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation


```bash
# Clone the repository
git clone https://github.com/your-username/cloud-cost-intelligence-retail.git
cd cloud-cost-intelligence-retail

# Install dependencies
pip install streamlit pandas numpy

# Run the app
streamlit run cloud_cost_combined_1.py
