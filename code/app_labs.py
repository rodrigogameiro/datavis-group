import streamlit as st
import pandas as pd
import plotly.express as px
import gzip
import os
from pathlib import Path

# Page configuration
st.set_page_config(page_title="MIMIC-IV Data Analysis", layout="wide")

st.title("MIMIC-IV Data Analysis Platform")

# ============================================================================
# Data loading functions
# ============================================================================
def get_data_path(relative_path):
    """Get data file path"""
    if os.path.exists(relative_path):
        return relative_path
    parent_path = os.path.join("..", relative_path)
    if os.path.exists(parent_path):
        return parent_path
    project_root = Path(__file__).parent.parent
    abs_path = project_root / relative_path
    if abs_path.exists():
        return str(abs_path)
    return relative_path

@st.cache_data
def load_labevents():
    """Load lab events data"""
    base_path = get_data_path("data/labevents.csv")
    gz_path = get_data_path("data/labevents.csv.gz")
    
    try:
        if os.path.exists(base_path):
            df = pd.read_csv(base_path)
        elif os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                df = pd.read_csv(f)
        else:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Failed to load lab data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_d_labitems():
    """Load lab items dictionary"""
    base_path = get_data_path("data/d_labitems.csv")
    gz_path = get_data_path("data/d_labitems.csv.gz")
    
    try:
        if os.path.exists(base_path):
            return pd.read_csv(base_path)
        elif os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                return pd.read_csv(f)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def load_pharmacy():
    """Load pharmacy/medication data"""
    base_path = get_data_path("data/pharmacy.csv")
    gz_path = get_data_path("data/pharmacy.csv.gz")
    
    try:
        if os.path.exists(base_path):
            return pd.read_csv(base_path)
        elif os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                return pd.read_csv(f)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def load_procedures():
    """Load procedures data"""
    gz_path = get_data_path("data/procedures_icd.csv.gz")
    
    try:
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                return pd.read_csv(f)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def load_d_procedures():
    """Load procedures dictionary"""
    gz_path = get_data_path("data/d_icd_procedures.csv.gz")
    
    try:
        if os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                return pd.read_csv(f)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data
def load_admissions():
    """Load admissions data"""
    base_path = get_data_path("data/admissions.csv")
    gz_path = get_data_path("data/admissions.csv.gz")
    
    try:
        if os.path.exists(base_path):
            return pd.read_csv(base_path)
        elif os.path.exists(gz_path):
            with gzip.open(gz_path, "rt") as f:
                return pd.read_csv(f)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ============================================================================
# Load data
# ============================================================================
with st.spinner("Loading data..."):
    labevents = load_labevents()
    d_labitems = load_d_labitems()
    pharmacy = load_pharmacy()
    procedures = load_procedures()
    d_procedures = load_d_procedures()
    admissions = load_admissions()

# Merge data
if not labevents.empty and not d_labitems.empty:
    labevents_merged = labevents.merge(d_labitems[["itemid", "label", "category", "fluid"]], 
                                       on="itemid", how="left")
else:
    labevents_merged = pd.DataFrame()

if not procedures.empty and not d_procedures.empty:
    procedures_merged = procedures.merge(d_procedures[["icd_code", "long_title"]], 
                                        on="icd_code", how="left")
else:
    procedures_merged = pd.DataFrame()

# ============================================================================
# Section 1: Lab Events
# ============================================================================
st.header("Lab Events")

if labevents_merged.empty:
    st.info("No lab events data available")
else:
    # Define available visualization options
    lab_options = {
        "Category": "category",
        "Fluid Type": "fluid",
        "Flag": "flag",
        "Priority": "priority",
        "Top Lab Items": "label"
    }
    
    # Select visualizations to display
    selected_lab = st.multiselect(
        "Choose visualizations to display",
        options=list(lab_options.keys()),
        default=["Category", "Fluid Type"],
        key="lab_select"
    )
    
    # Display charts based on selection
    if selected_lab:
        # Calculate number of columns
        num_cols = min(len(selected_lab), 2)
        cols = st.columns(num_cols)
        
        for idx, option in enumerate(selected_lab):
            col_idx = idx % num_cols
            field = lab_options[option]
            
            with cols[col_idx]:
                if option == "Top Lab Items":
                    # Display top 15 items
                    top_items = labevents_merged[field].value_counts().head(15).reset_index()
                    top_items.columns = [field, "count"]
                    fig = px.bar(top_items, x="count", y=field, orientation='h',
                                labels={field: "Lab Item", "count": "Count"},
                                title=option)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                else:
                    # Display bar chart or pie chart for other fields
                    counts = labevents_merged[field].fillna("Unknown").value_counts().head(10).reset_index()
                    counts.columns = [field, "count"]
                    
                    # Show pie chart if categories <= 5, otherwise show bar chart
                    if len(counts) <= 5:
                        fig = px.pie(counts, values="count", names=field,
                                    title=option)
                    else:
                        fig = px.bar(counts, x=field, y="count",
                                    labels={field: option, "count": "Count"},
                                    title=option)
                        fig.update_xaxes(tickangle=-45)
                
                st.plotly_chart(fig, width='stretch', key=f"lab_{option}")

st.divider()

# ============================================================================
# Section 2: Medication
# ============================================================================
st.header("Medication")

if pharmacy.empty:
    st.info("No medication data available")
else:
    # Define available visualization options
    med_options = {
        "Top Medications": "medication",
        "Status": "status",
        "Route": "route",
        "Procedure Type": "proc_type",
        "Frequency": "frequency"
    }
    
    # Select visualizations to display
    selected_med = st.multiselect(
        "Choose visualizations to display",
        options=list(med_options.keys()),
        default=["Top Medications", "Status"],
        key="med_select"
    )
    
    # Display charts based on selection
    if selected_med:
        # Calculate number of columns
        num_cols = min(len(selected_med), 2)
        cols = st.columns(num_cols)
        
        for idx, option in enumerate(selected_med):
            col_idx = idx % num_cols
            field = med_options[option]
            
            with cols[col_idx]:
                if option == "Top Medications":
                    # Display top 15 items
                    top_meds = pharmacy[field].value_counts().head(15).reset_index()
                    top_meds.columns = [field, "count"]
                    fig = px.bar(top_meds, x=field, y="count",
                                labels={field: "Medication", "count": "Count"},
                                title=option)
                    fig.update_xaxes(tickangle=-45)
                elif option == "Frequency":
                    # Display top 15 items, horizontal bar chart
                    freq_counts = pharmacy[field].fillna("Unknown").value_counts().head(15).reset_index()
                    freq_counts.columns = [field, "count"]
                    fig = px.bar(freq_counts, x="count", y=field, orientation='h',
                                labels={field: "Frequency", "count": "Count"},
                                title=option)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                else:
                    # Display bar chart or pie chart for other fields
                    counts = pharmacy[field].fillna("Unknown").value_counts().head(10).reset_index()
                    counts.columns = [field, "count"]
                    
                    # Show pie chart if categories <= 5, otherwise show bar chart
                    if len(counts) <= 5:
                        fig = px.pie(counts, values="count", names=field,
                                    title=option)
                    else:
                        fig = px.bar(counts, x=field, y="count",
                                    labels={field: option, "count": "Count"},
                                    title=option)
                        fig.update_xaxes(tickangle=-45)
                
                st.plotly_chart(fig, width='stretch', key=f"med_{option}")

st.divider()

# ============================================================================
# Section 3: Procedure
# ============================================================================
st.header("Procedure")

if procedures_merged.empty:
    st.info("No procedure data available")
else:
    # Define available visualization options
    proc_options = {
        "Top Procedures": "long_title",
        "Hospital Visits with/without Procedures": "procedure_status"
    }
    
    # Select visualizations to display
    selected_proc = st.multiselect(
        "Choose visualizations to display",
        options=list(proc_options.keys()),
        default=["Top Procedures", "Hospital Visits with/without Procedures"],
        key="proc_select"
    )
    
    # Display charts based on selection
    if selected_proc:
        # Calculate number of columns
        num_cols = min(len(selected_proc), 2)
        cols = st.columns(num_cols)
        
        for idx, option in enumerate(selected_proc):
            col_idx = idx % num_cols
            
            with cols[col_idx]:
                if option == "Top Procedures":
                    # Display top 15 items, horizontal bar chart
                    field = proc_options[option]
                    top_procedures = procedures_merged[field].value_counts().head(15).reset_index()
                    top_procedures.columns = [field, "count"]
                    fig = px.bar(top_procedures, x="count", y=field, orientation='h',
                                labels={field: "Procedure", "count": "Count"},
                                title=option)
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                    st.plotly_chart(fig, width='stretch', key=f"proc_{option}")
                elif option == "Hospital Visits with/without Procedures":
                    # Calculate hospital visits with and without procedures
                    if not admissions.empty and not procedures.empty:
                        # Get unique hadm_id from procedures
                        hadm_with_procedures = set(procedures["hadm_id"].dropna().unique())
                        # Get all hadm_id from admissions
                        all_hadm = set(admissions["hadm_id"].dropna().unique())
                        # Calculate counts
                        with_procedures = len(hadm_with_procedures)
                        without_procedures = len(all_hadm - hadm_with_procedures)
                        
                        # Create bar chart
                        visit_data = pd.DataFrame({
                            "Status": ["With Procedures", "Without Procedures"],
                            "Count": [with_procedures, without_procedures]
                        })
                        
                        fig = px.bar(visit_data, x="Status", y="Count",
                                    labels={"Status": "Hospital Visit Status", "Count": "Number of Visits"},
                                    title="Hospital Visits with/without Procedures",
                                    color="Status",
                                    color_discrete_map={"With Procedures": "#1f77b4", "Without Procedures": "#ff7f0e"})
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, width='stretch', key=f"proc_{option}")
                    else:
                        st.info("Admissions or procedures data not available")
