# 🏥 MIMIC-Explorer

**Interactive Exploration of the MIMIC-IV Clinical Database**

A Streamlit dashboard for exploring diagnosis patterns, patient outcomes, and clinical data in the MIMIC-IV Demo dataset — no coding required.

> Developed for **BMI 706: Data Visualization for Biomedical Applications**  
> Harvard Medical School | December 2025



---

## 🎯 Overview

New researchers often spend weeks understanding MIMIC's complex structure before conducting actual research. MIMIC-Explorer provides an intuitive GUI for:

- **Orienting** to the dataset (demographics, patient flow)
- **Exploring** diagnosis distributions and co-occurrence patterns
- **Analyzing** outcomes (mortality, length of stay, ICU)
- **Deep-diving** into specific diagnoses with comparison tools


## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- MIMIC-IV Demo data files (see [Data Setup](#data-setup))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mimic-explorer.git
cd mimic-explorer

# Create conda environment
conda create -n mimic-explorer python=3.11
conda activate mimic-explorer

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download the [MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/) from PhysioNet
2. Place CSV files in the `data/` directory:
   ```
   data/
   ├── patients.csv
   ├── admissions.csv
   ├── diagnoses_icd.csv
   ├── d_icd_diagnoses.csv
   ├── labevents.csv
   ├── d_labitems.csv
   ├── prescriptions.csv
   ├── pharmacy.csv
   ├── icustays.csv
   ├── procedures_icd.csv
   └── d_icd_procedures.csv
   ```

### Run the App

```bash
cd code
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🗂️ Project Structure

```
mimic-explorer/
├── code/
│   └── app.py              # Main Streamlit application
├── data/                   # MIMIC-IV Demo CSV 
├── requirements.txt
└── README.md
```

## 📊 Features

| Tab | Purpose | Key Visualizations |
|-----|---------|-------------------|
| **📊 Overview** | Dataset orientation | Metrics, demographics, Sankey flow, Pathway Explorer |
| **🩺 Diagnoses** | Distribution patterns | Top diagnoses (clickable), small multiples, co-occurrence heatmap |
| **🧪 Clinical Data** | Data inventory | Labs by category, top medications, procedures |
| **📈 Outcomes** | Outcome analysis | Survivors vs non-survivors, LOS, mortality by demographics |
| **🔬 Deep Dive** | Detailed exploration | Single diagnosis stats, comparison mode, contingency tables |

### Linked Views
Click any diagnosis bar in the **Diagnoses** tab → automatically populates the **Deep Dive** tab for detailed analysis.

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — Web framework
- **[Altair](https://altair-viz.github.io/)** — Declarative visualizations
- **[Plotly](https://plotly.com/)** — Sankey diagram
- **[Pandas](https://pandas.pydata.org/)** — Data manipulation

## 👥 Team

Douglas Jiang. Rodrigo Gameiro, Wanyan Yuan, Yuan Tian

## 📄 License

This project is for educational purposes as part of Harvard's BMI 706 course.

MIMIC-IV data is subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/2.2/).

## 🙏 Acknowledgments

- [MIMIC-IV](https://mimic.mit.edu/) team at MIT Lab for Computational Physiology
- BMI 706 course staff at Harvard Medical School

---

<p align="center">
  <i>Built with ❤️ for better clinical data exploration</i>
</p>
