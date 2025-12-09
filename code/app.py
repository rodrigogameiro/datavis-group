"""
================================================================================
MIMIC-Explorer: Consolidated Interactive Dashboard
================================================================================

An interactive Streamlit dashboard for exploring the MIMIC-IV clinical database.
Developed for Harvard Data Visualization for Biomedical Applications course.

Authors: Douglas Jiang, Rodrigo Gameiro, Wanyan Yuan, Yuan Tian
Date: December 2025

Structure:
    SECTION 1: Imports & Page Configuration
    SECTION 2: Data Loading Functions
    SECTION 3: Data Preparation Functions
    SECTION 4: Visualization Helper Functions
    SECTION 5: Load & Prepare Data
    SECTION 6: App Header
    SECTION 7: Tab Container
        - Tab 1: Overview (Demographics, Patient Flow)
        - Tab 2: Diagnoses (Patterns, Co-occurrence, Complexity)
        - Tab 3: Clinical Data (Labs, Medications, Procedures)
        - Tab 4: Outcomes (Mortality, LOS, Comparisons)
        - Tab 5: Deep Dive (Single Diagnosis, Comparison Mode, Contingency Tables)
    SECTION 8: Data Preview (Development Tool)

To run:
    cd code/
    streamlit run app.py
================================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS & PAGE CONFIGURATION
# =============================================================================

import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import numpy as np
from itertools import combinations, product

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MIMIC-Explorer",
    page_icon="🏥",
    layout="wide"
)

# Disable Altair row limit for larger datasets
alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
# These variables persist across interactions and enable cross-tab communication

if 'selected_diagnosis' not in st.session_state:
    st.session_state.selected_diagnosis = None


# =============================================================================
# SECTION 2: DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """
    Load all required MIMIC-IV tables from CSV files.
    
    Returns:
        dict: Dictionary containing all DataFrames keyed by table name
    
    Tables loaded:
        - patients: Patient demographics (subject_id, gender, anchor_age, dod)
        - admissions: Hospital admissions (hadm_id, admission_type, etc.)
        - diagnoses: Diagnosis codes per admission (diagnoses_icd)
        - d_icd_diagnoses: Diagnosis code descriptions
        - labevents: Laboratory test results
        - d_labitems: Lab test descriptions
        - prescriptions: Medication prescriptions
        - pharmacy: Pharmacy dispensing records
        - icustays: ICU stay information
        - procedures: Procedure codes (procedures_icd)
        - d_icd_procedures: Procedure code descriptions
    """
    data_path = "../data/"
    
    # Core tables (required)
    patients = pd.read_csv(data_path + "patients.csv")
    admissions = pd.read_csv(
        data_path + "admissions.csv",
        parse_dates=["admittime", "dischtime", "deathtime"]
    )
    diagnoses = pd.read_csv(data_path + "diagnoses_icd.csv")
    d_icd_diagnoses = pd.read_csv(data_path + "d_icd_diagnoses.csv")
    
    # Lab data
    labevents = pd.read_csv(data_path + "labevents.csv")
    d_labitems = pd.read_csv(data_path + "d_labitems.csv")
    
    # Medication data
    prescriptions = pd.read_csv(data_path + "prescriptions.csv")
    pharmacy = pd.read_csv(data_path + "pharmacy.csv")
    
    # ICU data
    icustays = pd.read_csv(data_path + "icustays.csv")
    
    # Procedure data
    procedures = pd.read_csv(data_path + "procedures_icd.csv")
    d_icd_procedures = pd.read_csv(data_path + "d_icd_procedures.csv")
    
    # Parse date of death for mortality calculations
    if 'dod' in patients.columns:
        patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')
    
    return {
        'patients': patients,
        'admissions': admissions,
        'diagnoses': diagnoses,
        'd_icd_diagnoses': d_icd_diagnoses,
        'labevents': labevents,
        'd_labitems': d_labitems,
        'prescriptions': prescriptions,
        'pharmacy': pharmacy,
        'icustays': icustays,
        'procedures': procedures,
        'd_icd_procedures': d_icd_procedures
    }


# =============================================================================
# SECTION 3: DATA PREPARATION FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1 Demographics Preparation
# -----------------------------------------------------------------------------

def simplify_race(race_value):
    """
    Standardize race/ethnicity categories into simplified groups.
    
    Args:
        race_value: Raw race/ethnicity string from MIMIC
        
    Returns:
        str: Simplified category (White, Black, Hispanic/Latino, Asian, Other/Unknown)
    """
    if pd.isna(race_value):
        return "Other/Unknown"
    
    x = str(race_value).upper()
    
    # Check for unknown/declined values first
    if any(k in x for k in ["UNKNOWN", "DECLINED", "UNABLE", "NOT SPECIFIED", "REFUSED"]):
        return "Other/Unknown"
    
    # Hispanic/Latino takes priority (can be any race)
    if "HISPANIC" in x or "LATINO" in x:
        return "Hispanic/Latino"
    
    # Then check specific races
    if "BLACK" in x or "AFRICAN" in x:
        return "Black"
    if "ASIAN" in x:
        return "Asian"
    if "WHITE" in x or "PORTUGUESE" in x:
        return "White"
    
    return "Other/Unknown"


def simplify_admission_location(location):
    """Simplify admission location into major categories."""
    if pd.isna(location):
        return "Unknown"
    
    x = str(location).upper()
    
    if "EMERGENCY" in x:
        return "Emergency"
    if any(k in x for k in ["REFERRAL", "WALK-IN", "SELF", "PROCEDURE"]):
        return "Referral"
    if "TRANSFER" in x:
        return "Transfer"
    if "PACU" in x:
        return "PACU"
    
    return "Other"


def simplify_admission_type(admission_type):
    """Simplify admission type into major categories."""
    if pd.isna(admission_type):
        return "Other"
    
    x = str(admission_type).upper()
    
    if "EMER" in x or "URGENT" in x:
        return "Emergency/Urgent"
    if "OBSERVATION" in x:
        return "Observation"
    if "SURGICAL SAME DAY" in x or "ELECTIVE" in x:
        return "Elective/Scheduled"
    
    return "Other"


def simplify_discharge_location(location):
    """Simplify discharge location into major categories."""
    if pd.isna(location):
        return "Other/Unknown"
    
    x = str(location).upper()
    
    if "DIED" in x or "HOSPICE" in x:
        return "Death/Hospice"
    if "HOME" in x or "AGAINST ADVICE" in x:
        return "Home/Community"
    if "SKILLED NURSING" in x or "REHAB" in x:
        return "Skilled Nursing/Rehab"
    if any(k in x for k in ["CHRONIC", "LONG TERM", "ACUTE HOSPITAL", "PSYCH"]):
        return "Other Facility"
    
    return "Other/Unknown"


@st.cache_data
def prepare_demographics(admissions):
    """
    Add simplified demographic columns to admissions DataFrame.
    
    Args:
        admissions: Raw admissions DataFrame
        
    Returns:
        DataFrame: Admissions with added simplified columns
    """
    df = admissions.copy()
    
    df['race_simplified'] = df['race'].apply(simplify_race)
    df['admission_loc_simple'] = df['admission_location'].apply(simplify_admission_location)
    df['admission_type_simple'] = df['admission_type'].apply(simplify_admission_type)
    df['discharge_loc_simple'] = df['discharge_location'].apply(simplify_discharge_location)
    df['marital_status'] = df['marital_status'].fillna('Unknown').replace({'': 'Unknown'})
    
    return df


# -----------------------------------------------------------------------------
# 3.2 Diagnosis Data Preparation
# -----------------------------------------------------------------------------

@st.cache_data
def prepare_diagnosis_data(_diagnoses, _d_icd):
    """
    Merge diagnoses with descriptions and compute patient counts.
    
    Args:
        _diagnoses: diagnoses_icd DataFrame
        _d_icd: d_icd_diagnoses DataFrame
        
    Returns:
        tuple: (dx_with_names, dx_counts)
            - dx_with_names: Diagnoses merged with descriptions
            - dx_counts: Diagnosis counts sorted by frequency
    """
    dx_with_names = _diagnoses.merge(
        _d_icd,
        on=['icd_code', 'icd_version'],
        how='left'
    )
    
    dx_counts = (
        dx_with_names
        .groupby(['icd_code', 'icd_version', 'long_title'])['subject_id']
        .nunique()
        .reset_index()
        .rename(columns={'subject_id': 'patient_count'})
        .sort_values('patient_count', ascending=False)
    )
    
    return dx_with_names, dx_counts


@st.cache_data
def get_diagnosis_patient_data(_diagnoses, _d_icd, _patients):
    """
    Create merged dataset linking diagnoses to patient demographics.
    
    Args:
        _diagnoses: diagnoses_icd DataFrame
        _d_icd: d_icd_diagnoses DataFrame
        _patients: patients DataFrame
        
    Returns:
        DataFrame: Diagnoses with patient demographic information
    """
    dx_with_names = _diagnoses.merge(
        _d_icd,
        on=['icd_code', 'icd_version'],
        how='left'
    )
    
    dx_with_patients = dx_with_names.merge(
        _patients[['subject_id', 'gender', 'anchor_age']],
        on='subject_id',
        how='left'
    )
    
    return dx_with_patients


@st.cache_data
def calculate_diagnosis_burden(_diagnoses):
    """
    Calculate how many unique diagnoses each patient has.
    
    Args:
        _diagnoses: diagnoses_icd DataFrame
        
    Returns:
        DataFrame: subject_id and diagnosis_count columns
    """
    burden = (
        _diagnoses
        .groupby('subject_id')['icd_code']
        .nunique()
        .reset_index()
        .rename(columns={'icd_code': 'diagnosis_count'})
    )
    return burden


@st.cache_data
def prepare_cooccurrence_matrix(_dx_with_names, _dx_counts, top_n=10):
    """
    Build co-occurrence matrix for heatmap visualization.
    
    Args:
        _dx_with_names: Diagnoses with descriptions
        _dx_counts: Diagnosis counts
        top_n: Number of top diagnoses to include
        
    Returns:
        tuple: (cooccurrence_df, top_dx_codes)
    """
    top_diagnoses = _dx_counts.head(top_n)['icd_code'].tolist()
    dx_filtered = _dx_with_names[_dx_with_names['icd_code'].isin(top_diagnoses)]
    patient_dx = dx_filtered.groupby('subject_id')['icd_code'].apply(set).reset_index()
    
    # Count co-occurrences
    cooccurrence = {}
    for _, row in patient_dx.iterrows():
        dx_list = list(row['icd_code'])
        for dx in dx_list:
            cooccurrence[(dx, dx)] = cooccurrence.get((dx, dx), 0) + 1
        for dx1, dx2 in combinations(dx_list, 2):
            cooccurrence[(dx1, dx2)] = cooccurrence.get((dx1, dx2), 0) + 1
            cooccurrence[(dx2, dx1)] = cooccurrence.get((dx2, dx1), 0) + 1
    
    # Convert to DataFrame
    records = [{'icd_code_1': dx1, 'icd_code_2': dx2, 'count': count} 
               for (dx1, dx2), count in cooccurrence.items()]
    cooccurrence_df = pd.DataFrame(records)
    
    # Ensure all pairs exist (fill missing with 0)
    all_pairs = pd.DataFrame(
        list(product(top_diagnoses, top_diagnoses)),
        columns=['icd_code_1', 'icd_code_2']
    )
    cooccurrence_df = all_pairs.merge(cooccurrence_df, on=['icd_code_1', 'icd_code_2'], how='left')
    cooccurrence_df['count'] = cooccurrence_df['count'].fillna(0).astype(int)
    
    # Add diagnosis names
    code_to_name = dict(zip(_dx_counts['icd_code'], _dx_counts['long_title']))
    cooccurrence_df['diagnosis_1'] = cooccurrence_df['icd_code_1'].map(code_to_name)
    cooccurrence_df['diagnosis_2'] = cooccurrence_df['icd_code_2'].map(code_to_name)
    cooccurrence_df['short_name_1'] = cooccurrence_df['diagnosis_1'].apply(
        lambda x: x[:20] + '...' if len(str(x)) > 20 else x
    )
    cooccurrence_df['short_name_2'] = cooccurrence_df['diagnosis_2'].apply(
        lambda x: x[:20] + '...' if len(str(x)) > 20 else x
    )
    
    return cooccurrence_df, top_diagnoses


@st.cache_data
def get_age_by_top_diagnoses(_dx_with_patients, _dx_counts, top_n=6):
    """
    Get age data for patients with each of the top N diagnoses.
    Used for small multiples visualization.
    
    Args:
        _dx_with_patients: Diagnoses with patient demographics
        _dx_counts: Diagnosis counts
        top_n: Number of diagnoses to include
        
    Returns:
        DataFrame: Age data for faceted visualization
    """
    top_dx_codes = _dx_counts.head(top_n)['icd_code'].tolist()
    
    age_by_dx = (
        _dx_with_patients[_dx_with_patients['icd_code'].isin(top_dx_codes)]
        [['subject_id', 'icd_code', 'long_title', 'anchor_age']]
        .drop_duplicates()
    )
    
    age_by_dx['short_title'] = age_by_dx['long_title'].apply(
        lambda x: x[:25] + '...' if len(str(x)) > 25 else x
    )
    
    # Add patient counts for sorting
    dx_patient_counts = age_by_dx.groupby('icd_code')['subject_id'].nunique().reset_index()
    dx_patient_counts.columns = ['icd_code', 'dx_patient_count']
    age_by_dx = age_by_dx.merge(dx_patient_counts, on='icd_code')
    
    return age_by_dx


# -----------------------------------------------------------------------------
# 3.3 Outcome Data Preparation
# -----------------------------------------------------------------------------

@st.cache_data
def get_outcome_comparison_data(_patients, _admissions, _diagnosis_burden):
    """
    Prepare comparison data for survivors vs non-survivors analysis.
    
    Args:
        _patients: patients DataFrame
        _admissions: admissions DataFrame
        _diagnosis_burden: diagnosis burden DataFrame
        
    Returns:
        DataFrame: Patient data with outcome and burden information
    """
    # Determine if patient died during any admission
    patient_outcomes = _admissions.groupby('subject_id')['hospital_expire_flag'].max().reset_index()
    patient_outcomes.columns = ['subject_id', 'died']
    
    # Merge with patient data
    comparison_df = _patients.merge(patient_outcomes, on='subject_id', how='left')
    comparison_df['died'] = comparison_df['died'].fillna(0).astype(int)
    comparison_df['outcome'] = comparison_df['died'].map({0: 'Survived', 1: 'Died'})
    comparison_df = comparison_df.merge(_diagnosis_burden, on='subject_id', how='left')
    
    return comparison_df


@st.cache_data
def build_patient_flow_data(_admissions, _icustays):
    """
    Build data for patient flow Sankey diagram.
    
    Flow: Admission Type → ICU Status → Outcome
    
    Args:
        _admissions: admissions DataFrame
        _icustays: icustays DataFrame
        
    Returns:
        tuple: (flow_data, node_list, node_indices)
    """
    flow_df = _admissions[['hadm_id', 'subject_id', 'admission_type', 'hospital_expire_flag']].copy()
    
    # Determine ICU status
    icu_hadm_ids = _icustays['hadm_id'].unique()
    flow_df['icu_status'] = flow_df['hadm_id'].isin(icu_hadm_ids).map({True: 'ICU Stay', False: 'No ICU'})
    flow_df['outcome'] = flow_df['hospital_expire_flag'].map({0: 'Survived', 1: 'Died'})
    
    # Group small admission types
    admission_counts = flow_df['admission_type'].value_counts()
    top_admissions = admission_counts[admission_counts >= 5].index.tolist()
    flow_df['admission_group'] = flow_df['admission_type'].apply(
        lambda x: x if x in top_admissions else 'Other'
    )
    
    # Create flow connections
    flow1 = flow_df.groupby(['admission_group', 'icu_status']).size().reset_index(name='count')
    flow1.columns = ['source', 'target', 'value']
    
    flow2 = flow_df.groupby(['icu_status', 'outcome']).size().reset_index(name='count')
    flow2.columns = ['source', 'target', 'value']
    
    all_flows = pd.concat([flow1, flow2], ignore_index=True)
    
    # Create node indices
    all_nodes = list(pd.concat([all_flows['source'], all_flows['target']]).unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    all_flows['source_idx'] = all_flows['source'].map(node_indices)
    all_flows['target_idx'] = all_flows['target'].map(node_indices)
    
    return all_flows, all_nodes, node_indices


# -----------------------------------------------------------------------------
# 3.4 Clinical Data Preparation
# -----------------------------------------------------------------------------

@st.cache_data
def get_top_labs_for_patients(_labevents, _d_labitems, patient_ids, top_n=10):
    """Get most common lab tests for a set of patients."""
    labs_filtered = _labevents[_labevents['subject_id'].isin(patient_ids)]
    lab_counts = (
        labs_filtered
        .merge(_d_labitems[['itemid', 'label']], on='itemid', how='left')
        .groupby('label')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
    )
    return lab_counts


@st.cache_data
def get_top_meds_for_patients(_prescriptions, patient_ids, top_n=10):
    """Get most common medications for a set of patients."""
    meds_filtered = _prescriptions[_prescriptions['subject_id'].isin(patient_ids)]
    med_counts = (
        meds_filtered
        .groupby('drug')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
    )
    return med_counts


@st.cache_data
def get_comorbidities(_dx_with_names, patient_ids, exclude_icd_code, top_n=10):
    """Get most common OTHER diagnoses for a set of patients."""
    comorbidities = (
        _dx_with_names[
            (_dx_with_names['subject_id'].isin(patient_ids)) & 
            (_dx_with_names['icd_code'] != exclude_icd_code)
        ]
        .groupby(['icd_code', 'long_title'])['subject_id']
        .nunique()
        .reset_index(name='patient_count')
        .sort_values('patient_count', ascending=False)
        .head(top_n)
    )
    return comorbidities


@st.cache_data
def get_comorbidities_for_comparison(_dx_with_names, patient_ids, exclude_icd_codes, top_n=10):
    """Get comorbidities excluding multiple diagnoses (for comparison mode)."""
    comorbidities = (
        _dx_with_names[
            (_dx_with_names['subject_id'].isin(patient_ids)) & 
            (~_dx_with_names['icd_code'].isin(exclude_icd_codes))
        ]
        .groupby(['icd_code', 'long_title'])['subject_id']
        .nunique()
        .reset_index(name='patient_count')
        .sort_values('patient_count', ascending=False)
        .head(top_n)
    )
    return comorbidities


@st.cache_data
def get_lab_distributions(_labevents, _d_labitems, patient_ids, top_n=6):
    """
    Get lab value distributions with reference range information.
    
    Args:
        _labevents: labevents DataFrame
        _d_labitems: d_labitems DataFrame
        patient_ids: Set of patient IDs to filter
        top_n: Number of top labs to return
        
    Returns:
        tuple: (lab_data, ref_ranges, top_labs)
    """
    labs_filtered = _labevents[
        (_labevents['subject_id'].isin(patient_ids)) & 
        (_labevents['valuenum'].notna())
    ].copy()
    
    labs_filtered = labs_filtered.merge(_d_labitems[['itemid', 'label']], on='itemid', how='left')
    
    # Get top labs by frequency
    top_labs = (
        labs_filtered.groupby('label')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)['label'].tolist()
    )
    
    lab_data = labs_filtered[labs_filtered['label'].isin(top_labs)].copy()
    
    # Calculate reference range statistics
    ref_ranges = (
        lab_data.groupby('label')
        .agg({
            'ref_range_lower': 'median',
            'ref_range_upper': 'median',
            'valuenum': ['median', 'mean', 'std', 'count']
        })
        .reset_index()
    )
    ref_ranges.columns = ['label', 'ref_lower', 'ref_upper', 'median_val', 'mean_val', 'std_val', 'n_measurements']
    
    # Calculate abnormal percentages
    lab_data = lab_data.merge(ref_ranges[['label', 'ref_lower', 'ref_upper']], on='label', how='left')
    lab_data['is_abnormal'] = (
        (lab_data['valuenum'] < lab_data['ref_lower']) | 
        (lab_data['valuenum'] > lab_data['ref_upper'])
    )
    
    abnormal_pct = (
        lab_data.groupby('label')['is_abnormal']
        .mean()
        .reset_index()
        .rename(columns={'is_abnormal': 'pct_abnormal'})
    )
    
    ref_ranges = ref_ranges.merge(abnormal_pct, on='label', how='left')
    
    return lab_data, ref_ranges, top_labs


# =============================================================================
# SECTION 4: VISUALIZATION HELPER FUNCTIONS
# =============================================================================

def smart_truncate(text, max_length=40):
    """
    Truncate text at word boundary, not mid-word.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        
    Returns:
        str: Truncated text with '...' if needed
    """
    text = str(text)
    if len(text) <= max_length:
        return text
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + '...'


def get_diagnosis_stats(dx_name, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden):
    """
    Get comprehensive statistics for a single diagnosis.
    Used in both single diagnosis mode and comparison mode.
    
    Args:
        dx_name: Full diagnosis name (long_title)
        dx_counts: Diagnosis counts DataFrame
        dx_with_names: Diagnoses with descriptions
        patients: patients DataFrame
        admissions: admissions DataFrame
        icustays: icustays DataFrame
        diagnosis_burden: diagnosis burden DataFrame
        
    Returns:
        dict: Comprehensive statistics for the diagnosis
    """
    icd_code = dx_counts[dx_counts['long_title'] == dx_name]['icd_code'].values[0]
    patient_ids = dx_with_names[dx_with_names['icd_code'] == icd_code]['subject_id'].unique()
    patient_data = patients[patients['subject_id'].isin(patient_ids)]
    
    # Mortality (admission-based)
    dx_admissions = admissions[admissions['subject_id'].isin(patient_ids)]
    mortality = dx_admissions['hospital_expire_flag'].mean() * 100 if len(dx_admissions) > 0 else 0
    
    # ICU rate (admission-based for consistency with Sankey)
    hadm_ids = dx_with_names[dx_with_names['icd_code'] == icd_code]['hadm_id'].unique()
    icu_hadm_ids = set(icustays['hadm_id'].unique())
    icu_rate = len(set(hadm_ids) & icu_hadm_ids) / len(hadm_ids) * 100 if len(hadm_ids) > 0 else 0
    
    # Demographics
    avg_age = patient_data['anchor_age'].mean() if len(patient_data) > 0 else 0
    pct_female = (patient_data['gender'] == 'F').mean() * 100 if len(patient_data) > 0 else 0
    
    # Diagnosis burden
    patient_burden = diagnosis_burden[diagnosis_burden['subject_id'].isin(patient_ids)]
    median_burden = patient_burden['diagnosis_count'].median() if len(patient_burden) > 0 else 0
    
    return {
        'icd_code': icd_code,
        'patient_ids': patient_ids,
        'patient_data': patient_data,
        'n_patients': len(patient_ids),
        'mortality': mortality,
        'icu_rate': icu_rate,
        'avg_age': avg_age,
        'pct_female': pct_female,
        'median_burden': median_burden,
        'hadm_ids': hadm_ids
    }


def create_bar_chart(data, x_col, y_col, title=None, color='#4C78A8', height=300, 
                     x_title=None, y_title=None, sort='-x'):
    """
    Create a standardized Altair horizontal bar chart.
    
    Args:
        data: DataFrame with data to plot
        x_col: Column for x-axis (typically count)
        y_col: Column for y-axis (typically category)
        title: Chart title
        color: Bar color
        height: Chart height in pixels
        x_title: X-axis title
        y_title: Y-axis title
        sort: Sort order for y-axis
        
    Returns:
        alt.Chart: Configured Altair chart
    """
    chart = alt.Chart(data).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:Q', title=x_title or x_col),
        y=alt.Y(f'{y_col}:N', title=y_title, sort=sort),
        tooltip=[
            alt.Tooltip(f'{y_col}:N', title=y_title or y_col),
            alt.Tooltip(f'{x_col}:Q', title=x_title or 'Count')
        ]
    ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


# =============================================================================
# SECTION 5: LOAD & PREPARE DATA
# =============================================================================

# Load all data tables
data = load_data()

# Unpack for convenience
patients = data['patients']
admissions = data['admissions']
diagnoses = data['diagnoses']
d_icd_diagnoses = data['d_icd_diagnoses']
labevents = data['labevents']
d_labitems = data['d_labitems']
prescriptions = data['prescriptions']
pharmacy = data['pharmacy']
icustays = data['icustays']
procedures = data['procedures']
d_icd_procedures = data['d_icd_procedures']

# Prepare derived datasets
admissions_demo = prepare_demographics(admissions)
dx_with_names, dx_counts = prepare_diagnosis_data(diagnoses, d_icd_diagnoses)
dx_with_patients = get_diagnosis_patient_data(diagnoses, d_icd_diagnoses, patients)
diagnosis_burden = calculate_diagnosis_burden(diagnoses)
cooccurrence_df, top_dx_codes = prepare_cooccurrence_matrix(dx_with_names, dx_counts, top_n=10)
age_by_top_dx = get_age_by_top_diagnoses(dx_with_patients, dx_counts, top_n=6)
outcome_comparison_df = get_outcome_comparison_data(patients, admissions, diagnosis_burden)
flow_data, flow_nodes, node_indices = build_patient_flow_data(admissions, icustays)

# Merge lab events with lab item descriptions
labevents_merged = labevents.merge(
    d_labitems[['itemid', 'label', 'category', 'fluid']], 
    on='itemid', 
    how='left'
)

# Merge procedures with descriptions
procedures_merged = procedures.merge(
    d_icd_procedures[['icd_code', 'long_title']], 
    on='icd_code', 
    how='left'
)

# Calculate overall statistics for reference
overall_mortality = admissions['hospital_expire_flag'].mean() * 100
overall_avg_age = patients['anchor_age'].mean()
overall_median_burden = diagnosis_burden['diagnosis_count'].median()


# =============================================================================
# SECTION 6: APP HEADER
# =============================================================================

st.title("🏥 MIMIC-Explorer")
st.markdown("### Interactive Exploration of the MIMIC-IV Clinical Database")
st.markdown(
    "*Navigate through the tabs below to explore patient demographics, diagnoses, "
    "clinical data, outcomes, and dive deep into specific conditions.*"
)

st.divider()


# =============================================================================
# SECTION 7: TAB CONTAINER
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🩺 Diagnoses", 
    "🧪 Clinical Data",
    "📈 Outcomes",
    "🔬 Deep Dive"
])


# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.header("📊 Dataset Overview")
    st.markdown("*Get oriented with the data: who are the patients and how do they flow through the system?*")
    
    # -------------------------------------------------------------------------
    # 1.1 At a Glance - Key Metrics
    # -------------------------------------------------------------------------
    st.subheader("🔢 At a Glance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients", len(patients))
    with col2:
        st.metric("Admissions", len(admissions))
    with col3:
        st.metric("Diagnosis Records", f"{len(diagnoses):,}")
    with col4:
        st.metric("Unique Diagnoses", diagnoses['icd_code'].nunique())
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 1.2 Demographics
    # -------------------------------------------------------------------------
    st.subheader("👥 Patient Demographics")
    st.markdown("*Who are the patients in this dataset?*")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        # Race distribution
        race_counts = admissions_demo['race_simplified'].value_counts().reset_index()
        race_counts.columns = ['race', 'count']
        
        race_chart = alt.Chart(race_counts).mark_bar().encode(
            x=alt.X('race:N', title='Race/Ethnicity', sort='-y',
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count:Q', title='Number of Admissions'),
            color=alt.Color('race:N', scale=alt.Scale(scheme='magma'), legend=None),
            tooltip=[alt.Tooltip('race:N', title='Race'), 
                    alt.Tooltip('count:Q', title='Admissions')]
        ).properties(title='Race/Ethnicity Distribution', height=300)
        
        st.altair_chart(race_chart, use_container_width=True)
    
    with demo_col2:
        # Insurance distribution
        insurance_counts = admissions_demo['insurance'].value_counts().reset_index()
        insurance_counts.columns = ['insurance', 'count']
        
        insurance_chart = alt.Chart(insurance_counts).mark_bar().encode(
            x=alt.X('insurance:N', title='Insurance', sort='-y',
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count:Q', title='Number of Admissions'),
            color=alt.Color('insurance:N', scale=alt.Scale(scheme='magma'), legend=None),
            tooltip=[alt.Tooltip('insurance:N', title='Insurance'), 
                    alt.Tooltip('count:Q', title='Admissions')]
        ).properties(title='Insurance Distribution', height=300)
        
        st.altair_chart(insurance_chart, use_container_width=True)
    
    demo_col3, demo_col4 = st.columns(2)
    
    with demo_col3:
        # Marital status
        marital_counts = admissions_demo['marital_status'].value_counts().reset_index()
        marital_counts.columns = ['status', 'count']
        
        marital_chart = alt.Chart(marital_counts).mark_bar().encode(
            x=alt.X('status:N', title='Marital Status', sort='-y',
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count:Q', title='Number of Admissions'),
            color=alt.Color('status:N', scale=alt.Scale(scheme='magma'), legend=None),
            tooltip=[alt.Tooltip('status:N', title='Status'), 
                    alt.Tooltip('count:Q', title='Admissions')]
        ).properties(title='Marital Status', height=250)
        
        st.altair_chart(marital_chart, use_container_width=True)
    
    with demo_col4:
        # Admission type
        admit_counts = admissions_demo['admission_type_simple'].value_counts().reset_index()
        admit_counts.columns = ['type', 'count']
        
        admit_chart = alt.Chart(admit_counts).mark_bar().encode(
            x=alt.X('type:N', title='Admission Type', sort='-y',
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count:Q', title='Number of Admissions'),
            color=alt.Color('type:N', scale=alt.Scale(scheme='magma'), legend=None),
            tooltip=[alt.Tooltip('type:N', title='Type'), 
                    alt.Tooltip('count:Q', title='Admissions')]
        ).properties(title='Admission Type', height=250)
        
        st.altair_chart(admit_chart, use_container_width=True)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 1.3 Patient Flow Sankey Diagram
    # -------------------------------------------------------------------------
    st.subheader("🌊 Patient Flow")
    st.markdown("*How do patients move through the hospital system? Follow the flow from admission to outcome.*")
    
    # Define node colors
    node_colors = []
    for node in flow_nodes:
        if node == 'Died':
            node_colors.append('#E45756')  # Red for mortality
        elif node == 'Survived':
            node_colors.append('#4C78A8')  # Blue for survival
        elif node == 'ICU Stay':
            node_colors.append('#F58518')  # Orange for ICU
        elif node == 'No ICU':
            node_colors.append('#72B7B2')  # Teal for no ICU
        else:
            node_colors.append('#B73779')  # Magma pink for admission types
    
    # Define link colors based on target
    link_colors = []
    for _, row in flow_data.iterrows():
        target = row['target']
        if target == 'Died':
            link_colors.append('rgba(228, 87, 86, 0.4)')
        elif target == 'Survived':
            link_colors.append('rgba(76, 120, 168, 0.4)')
        elif target == 'ICU Stay':
            link_colors.append('rgba(245, 133, 24, 0.4)')
        elif target == 'No ICU':
            link_colors.append('rgba(114, 183, 178, 0.4)')
        else:
            link_colors.append('rgba(183, 55, 121, 0.4)')
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=flow_nodes,
            color=node_colors
        ),
        link=dict(
            source=flow_data['source_idx'].tolist(),
            target=flow_data['target_idx'].tolist(),
            value=flow_data['value'].tolist(),
            color=link_colors
        )
    )])
    
    fig.update_layout(
        font=dict(size=12, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Width of each flow represents number of admissions. Red flows lead to mortality.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 1.4 Pathway Explorer
    # -------------------------------------------------------------------------
    st.subheader("🔎 Explore a Patient Pathway")
    st.markdown("*Select a path through the hospital to see who those patients are.*")
    
    # Build pathway filter data
    pathway_df = admissions[['hadm_id', 'subject_id', 'admission_type', 'hospital_expire_flag']].copy()
    icu_hadm_set = set(icustays['hadm_id'].unique())
    pathway_df['icu_status'] = pathway_df['hadm_id'].apply(lambda x: 'ICU Stay' if x in icu_hadm_set else 'No ICU')
    pathway_df['outcome'] = pathway_df['hospital_expire_flag'].map({0: 'Survived', 1: 'Died'})
    
    # Filter dropdowns
    admission_types = ['All'] + sorted(pathway_df['admission_type'].unique().tolist())
    icu_options = ['All', 'ICU Stay', 'No ICU']
    outcome_options = ['All', 'Survived', 'Died']
    
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        selected_admission = st.selectbox("Admission Type:", admission_types, key="pathway_admission")
    with pcol2:
        selected_icu = st.selectbox("ICU Status:", icu_options, key="pathway_icu")
    with pcol3:
        selected_outcome = st.selectbox("Outcome:", outcome_options, key="pathway_outcome")
    
    # Apply filters
    filtered_pathway = pathway_df.copy()
    if selected_admission != 'All':
        filtered_pathway = filtered_pathway[filtered_pathway['admission_type'] == selected_admission]
    if selected_icu != 'All':
        filtered_pathway = filtered_pathway[filtered_pathway['icu_status'] == selected_icu]
    if selected_outcome != 'All':
        filtered_pathway = filtered_pathway[filtered_pathway['outcome'] == selected_outcome]
    
    pathway_patient_ids = filtered_pathway['subject_id'].unique()
    pathway_patients = patients[patients['subject_id'].isin(pathway_patient_ids)]
    
    n_admissions = len(filtered_pathway)
    n_patients = len(pathway_patient_ids)
    
    st.info(f"📌 **{n_admissions} admissions** from **{n_patients} patients** match this pathway")
    
    if n_patients > 0:
        # Demographics row
        pw_col1, pw_col2, pw_col3, pw_col4 = st.columns(4)
        with pw_col1:
            st.metric("Patients", n_patients)
        with pw_col2:
            avg_age = pathway_patients['anchor_age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f} yrs")
        with pw_col3:
            pct_female = (pathway_patients['gender'] == 'F').mean() * 100
            st.metric("% Female", f"{pct_female:.0f}%")
        with pw_col4:
            pathway_burden = diagnosis_burden[diagnosis_burden['subject_id'].isin(pathway_patient_ids)]
            med_burden = pathway_burden['diagnosis_count'].median() if len(pathway_burden) > 0 else 0
            st.metric("Median Diagnoses", f"{med_burden:.0f}")
        
        # Top diagnoses and medications
        pw_col1, pw_col2 = st.columns(2)
        
        with pw_col1:
            st.markdown("**Top Diagnoses in This Pathway**")
            pathway_dx = (
                dx_with_names[dx_with_names['subject_id'].isin(pathway_patient_ids)]
                .groupby('long_title')['subject_id'].nunique()
                .reset_index()
                .rename(columns={'subject_id': 'count'})
                .sort_values('count', ascending=False)
                .head(8)
            )
            pathway_dx['short_title'] = pathway_dx['long_title'].apply(lambda x: smart_truncate(x, 35))
            
            if len(pathway_dx) > 0:
                dx_chart = alt.Chart(pathway_dx).mark_bar(color='#72B7B2').encode(
                    x=alt.X('count:Q', title='Patients'),
                    y=alt.Y('short_title:N', title=None, sort='-x'),
                    tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), 
                            alt.Tooltip('count:Q', title='Patients')]
                ).properties(height=220)
                st.altair_chart(dx_chart, use_container_width=True)
        
        with pw_col2:
            st.markdown("**Top Medications in This Pathway**")
            pathway_meds = get_top_meds_for_patients(prescriptions, pathway_patient_ids).head(8)
            
            if len(pathway_meds) > 0:
                pathway_meds['short_drug'] = pathway_meds['drug'].apply(lambda x: smart_truncate(x, 20))
                med_chart = alt.Chart(pathway_meds).mark_bar(color='#72B7B2').encode(
                    x=alt.X('count:Q', title='Prescriptions'),
                    y=alt.Y('short_drug:N', title=None, sort='-x'),
                    tooltip=[alt.Tooltip('drug:N', title='Medication'), 
                            alt.Tooltip('count:Q', title='Prescriptions')]
                ).properties(height=220)
                st.altair_chart(med_chart, use_container_width=True)
            else:
                st.info("No medication data for this pathway.")
    else:
        st.warning("No patients match this pathway combination.")


# -----------------------------------------------------------------------------
# TAB 2: DIAGNOSES
# -----------------------------------------------------------------------------
with tab2:
    st.header("🩺 Diagnosis Patterns")
    st.markdown("*Explore what conditions appear, how they relate, and patient complexity.*")
    
    # -------------------------------------------------------------------------
    # 2.1 Top Diagnoses Bar Chart
    # -------------------------------------------------------------------------
    st.subheader("🏆 Most Common Diagnoses")
    
    top_n = st.slider(
        "Number of diagnoses to show",
        min_value=5,
        max_value=25,
        value=10,
        help="Adjust how many top diagnoses appear in the bar chart"
    )
    
    st.markdown(f"Showing the **{top_n} most common diagnoses** by number of patients")
    
    top_dx = dx_counts.head(top_n).copy()
    top_dx['short_title'] = top_dx['long_title'].apply(smart_truncate)
    
    # Create clickable bar chart
    click_selection = alt.selection_point(fields=['long_title'])
    
    bars = alt.Chart(top_dx).mark_bar(cursor='pointer').encode(
        x=alt.X('patient_count:Q', title='Number of Patients'),
        y=alt.Y('short_title:N', title='Diagnosis', sort='-x'),
        color=alt.condition(
            click_selection,
            alt.value('#4C78A8'),
            alt.value('#E0E0E0')
        ),
        tooltip=[
            alt.Tooltip('long_title:N', title='Diagnosis'),
            alt.Tooltip('icd_code:N', title='ICD Code'),
            alt.Tooltip('patient_count:Q', title='Patients')
        ]
    ).properties(
        height=max(300, top_n * 28)
    ).add_params(click_selection)
    
    chart_event = st.altair_chart(bars, use_container_width=True, on_select="rerun")
    
    # Handle click events
    if chart_event and chart_event.selection and 'param_1' in chart_event.selection:
        selected_points = chart_event.selection['param_1']
        if selected_points and len(selected_points) > 0:
            clicked_diagnosis = selected_points[0].get('long_title')
            if clicked_diagnosis:
                st.session_state.selected_diagnosis = clicked_diagnosis
    
    st.caption("💡 Click any bar to select a diagnosis for detailed analysis in the Deep Dive tab.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 2.2 Age Patterns (Small Multiples)
    # -------------------------------------------------------------------------
    st.subheader("📊 Patterns Across Top Diagnoses")
    st.markdown("*How do age and outcomes vary across the most common conditions?*")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**Age Distribution by Diagnosis**")
        
        small_multiples = alt.Chart(age_by_top_dx).mark_bar(color='#B73779').encode(
            x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
            y=alt.Y('count()', title='Patients'),
            tooltip=[
                alt.Tooltip('long_title:N', title='Diagnosis'),
                alt.Tooltip('anchor_age:Q', bin=alt.Bin(step=10), title='Age Range'),
                alt.Tooltip('count()', title='Patients')
            ]
        ).properties(
            width=120,
            height=100
        ).facet(
            facet=alt.Facet('short_title:N', title=None,
                           sort=alt.EncodingSortField('dx_patient_count', order='descending')),
            columns=3
        ).resolve_scale(y='independent')
        
        st.altair_chart(small_multiples, use_container_width=False)
    
    with col2:
        st.markdown("**Mortality & ICU Rates**")
        
        # Calculate rates for top 6 diagnoses
        top_6_codes = dx_counts.head(6)['icd_code'].tolist()
        outcome_data = []
        
        for code in top_6_codes:
            dx_name = dx_counts[dx_counts['icd_code'] == code]['long_title'].values[0]
            short_name = smart_truncate(dx_name, 50)
            
            pts = dx_with_names[dx_with_names['icd_code'] == code]['subject_id'].unique()
            dx_admissions = admissions[admissions['subject_id'].isin(pts)]
            mortality = dx_admissions['hospital_expire_flag'].mean() * 100
            
            dx_admissions_ids = dx_with_names[dx_with_names['icd_code'] == code]['hadm_id'].unique()
            icu_hadm_ids = icustays['hadm_id'].unique()
            dx_icu_admissions = set(dx_admissions_ids) & set(icu_hadm_ids)
            icu_rate = len(dx_icu_admissions) / len(dx_admissions_ids) * 100 if len(dx_admissions_ids) > 0 else 0
            
            outcome_data.append({
                'diagnosis': short_name, 
                'Mortality %': mortality, 
                'ICU %': icu_rate
            })
        
        outcome_df = pd.DataFrame(outcome_data)
        dx_order = outcome_df['diagnosis'].tolist()
        
        # Mortality chart
        mortality_chart = alt.Chart(outcome_df).mark_bar(color='#E45756').encode(
            y=alt.Y('diagnosis:N', title=None, sort=dx_order),
            x=alt.X('Mortality %:Q', title='Mortality %'),
            tooltip=[alt.Tooltip('diagnosis:N', title='Diagnosis'), 
                    alt.Tooltip('Mortality %:Q', format='.1f')]
        ).properties(height=150, title='Mortality Rate')
        
        # ICU chart
        icu_chart = alt.Chart(outcome_df).mark_bar(color='#F58518').encode(
            y=alt.Y('diagnosis:N', title=None, sort=dx_order),
            x=alt.X('ICU %:Q', title='ICU %'),
            tooltip=[alt.Tooltip('diagnosis:N', title='Diagnosis'), 
                    alt.Tooltip('ICU %:Q', format='.1f')]
        ).properties(height=150, title='ICU Admission Rate')
        
        outcome_chart = alt.vconcat(mortality_chart, icu_chart).resolve_scale(x='independent')
        st.altair_chart(outcome_chart, use_container_width=True)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 2.3 Co-occurrence Heatmap
    # -------------------------------------------------------------------------
    st.subheader("🔗 Diagnosis Co-occurrence")
    st.markdown("*Which diagnoses frequently appear together? Brighter cells indicate stronger co-occurrence.*")
    
    sort_order = dx_counts[dx_counts['icd_code'].isin(top_dx_codes)]['long_title'].tolist()
    short_sort_order = [smart_truncate(x, 20) for x in sort_order]
    
    heatmap = alt.Chart(cooccurrence_df).mark_rect().encode(
        x=alt.X('short_name_1:N', title=None, sort=short_sort_order,
               axis=alt.Axis(labelAngle=-45, labelLimit=120, labelFontSize=11)),
        y=alt.Y('short_name_2:N', title=None, sort=short_sort_order),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='magma'), title='Patients'),
        tooltip=[
            alt.Tooltip('diagnosis_1:N', title='Diagnosis 1'),
            alt.Tooltip('diagnosis_2:N', title='Diagnosis 2'),
            alt.Tooltip('count:Q', title='Patients with both')
        ]
    ).properties(width=500, height=500)
    
    st.altair_chart(heatmap, use_container_width=True)
    st.caption("💡 Diagonal shows patients with each diagnosis. Off-diagonal shows co-occurrence.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 2.4 Patient Complexity (Diagnosis Burden)
    # -------------------------------------------------------------------------
    st.subheader("📈 Patient Complexity")
    st.markdown("*How many diagnoses do patients have? This reveals the complexity of the patient population.*")
    
    median_burden = diagnosis_burden['diagnosis_count'].median()
    mean_burden = diagnosis_burden['diagnosis_count'].mean()
    max_burden = diagnosis_burden['diagnosis_count'].max()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median Diagnoses", f"{median_burden:.0f}")
    with col2:
        st.metric("Average Diagnoses", f"{mean_burden:.1f}")
    with col3:
        st.metric("Maximum Diagnoses", f"{max_burden:.0f}")
    
    # Burden histogram
    burden_hist = alt.Chart(diagnosis_burden).mark_bar().encode(
        x=alt.X('diagnosis_count:Q', bin=alt.Bin(step=5), title='Number of Diagnoses per Patient'),
        y=alt.Y('count()', title='Number of Patients'),
        color=alt.value('#B73779'),
        tooltip=[
            alt.Tooltip('diagnosis_count:Q', bin=alt.Bin(step=5), title='Diagnosis Range'),
            alt.Tooltip('count()', title='Patients')
        ]
    ).properties(height=300)
    
    # Add median reference line
    median_line = alt.Chart(pd.DataFrame({'median': [median_burden]})).mark_rule(
        color='red', strokeDash=[5, 5], strokeWidth=2
    ).encode(x='median:Q')
    
    burden_chart = burden_hist + median_line
    st.altair_chart(burden_chart, use_container_width=True)
    st.caption(f"📊 Red dashed line = median ({median_burden:.0f} diagnoses). Note the right skew — some patients have 100+ diagnoses.")


# -----------------------------------------------------------------------------
# TAB 3: CLINICAL DATA
# -----------------------------------------------------------------------------
with tab3:
    st.header("🧪 Clinical Data Overview")
    st.markdown("*Explore what laboratory tests, medications, and procedures are captured in the dataset.*")
    
    # -------------------------------------------------------------------------
    # 3.1 Lab Events Overview
    # -------------------------------------------------------------------------
    st.subheader("🔬 Laboratory Tests")
    
    if len(labevents_merged) > 0:
        lab_col1, lab_col2 = st.columns(2)
        
        with lab_col1:
            st.markdown("**Lab Tests by Category**")
            category_counts = labevents_merged['category'].fillna('Unknown').value_counts().head(10).reset_index()
            category_counts.columns = ['category', 'count']
            
            cat_chart = alt.Chart(category_counts).mark_bar().encode(
                x=alt.X('category:N', title='Category', sort='-y', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('count:Q', title='Number of Tests'),
                color=alt.Color('category:N', scale=alt.Scale(scheme='magma'), legend=None),
                tooltip=[alt.Tooltip('category:N'), alt.Tooltip('count:Q', title='Tests')]
            ).properties(height=300)
            st.altair_chart(cat_chart, use_container_width=True)
        
        with lab_col2:
            st.markdown("**Lab Tests by Fluid Type**")
            fluid_counts = labevents_merged['fluid'].fillna('Unknown').value_counts().head(10).reset_index()
            fluid_counts.columns = ['fluid', 'count']
            
            fluid_chart = alt.Chart(fluid_counts).mark_bar().encode(
                x=alt.X('fluid:N', title='Fluid', sort='-y', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('count:Q', title='Number of Tests'),
                color=alt.Color('fluid:N', scale=alt.Scale(scheme='magma'), legend=None),
                tooltip=[alt.Tooltip('fluid:N'), alt.Tooltip('count:Q', title='Tests')]
            ).properties(height=300)
            st.altair_chart(fluid_chart, use_container_width=True)
        
        # Top lab items
        st.markdown("**Most Common Lab Tests**")
        top_labs = labevents_merged['label'].value_counts().head(15).reset_index()
        top_labs.columns = ['label', 'count']
        top_labs['short_label'] = top_labs['label'].apply(lambda x: smart_truncate(x, 30))
        
        top_lab_chart = alt.Chart(top_labs).mark_bar(color='#B73779').encode(
            x=alt.X('count:Q', title='Number of Tests'),
            y=alt.Y('short_label:N', title=None, sort='-x'),
            tooltip=[alt.Tooltip('label:N', title='Lab Test'), alt.Tooltip('count:Q', title='Tests')]
        ).properties(height=400)
        st.altair_chart(top_lab_chart, use_container_width=True)
    else:
        st.info("No lab events data available.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 3.2 Medications Overview
    # -------------------------------------------------------------------------
    st.subheader("💊 Medications")
    
    med_col1, med_col2 = st.columns(2)
    
    with med_col1:
        st.markdown("**Most Common Medications (Prescriptions)**")
        if len(prescriptions) > 0:
            top_meds = prescriptions['drug'].value_counts().head(15).reset_index()
            top_meds.columns = ['drug', 'count']
            top_meds['short_drug'] = top_meds['drug'].apply(lambda x: smart_truncate(x, 25))
            
            med_chart = alt.Chart(top_meds).mark_bar(color='#4C78A8').encode(
                x=alt.X('count:Q', title='Prescriptions'),
                y=alt.Y('short_drug:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('drug:N', title='Medication'), alt.Tooltip('count:Q', title='Prescriptions')]
            ).properties(height=400)
            st.altair_chart(med_chart, use_container_width=True)
        else:
            st.info("No prescription data available.")
    
    with med_col2:
        st.markdown("**Pharmacy Records by Status**")
        if len(pharmacy) > 0 and 'status' in pharmacy.columns:
            status_counts = pharmacy['status'].fillna('Unknown').value_counts().reset_index()
            status_counts.columns = ['status', 'count']
            
            status_chart = alt.Chart(status_counts).mark_bar().encode(
                x=alt.X('status:N', title='Status', sort='-y'),
                y=alt.Y('count:Q', title='Records'),
                color=alt.Color('status:N', scale=alt.Scale(scheme='magma'), legend=None),
                tooltip=[alt.Tooltip('status:N'), alt.Tooltip('count:Q', title='Records')]
            ).properties(height=300)
            st.altair_chart(status_chart, use_container_width=True)
        else:
            st.info("No pharmacy status data available.")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 3.3 Procedures Overview
    # -------------------------------------------------------------------------
    st.subheader("🏥 Procedures")
    
    proc_col1, proc_col2 = st.columns(2)
    
    with proc_col1:
        st.markdown("**Most Common Procedures**")
        if len(procedures_merged) > 0:
            top_procs = procedures_merged['long_title'].value_counts().head(15).reset_index()
            top_procs.columns = ['procedure', 'count']
            top_procs['short_proc'] = top_procs['procedure'].apply(lambda x: smart_truncate(str(x), 35))
            
            proc_chart = alt.Chart(top_procs).mark_bar(color='#72B7B2').encode(
                x=alt.X('count:Q', title='Count'),
                y=alt.Y('short_proc:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('procedure:N', title='Procedure'), alt.Tooltip('count:Q', title='Count')]
            ).properties(height=400)
            st.altair_chart(proc_chart, use_container_width=True)
        else:
            st.info("No procedure data available.")
    
    with proc_col2:
        st.markdown("**Admissions with Procedures**")
        if len(procedures) > 0:
            hadm_with_proc = set(procedures['hadm_id'].dropna().unique())
            all_hadm = set(admissions['hadm_id'].dropna().unique())
            
            proc_status = pd.DataFrame({
                'Status': ['With Procedures', 'Without Procedures'],
                'Count': [len(hadm_with_proc), len(all_hadm - hadm_with_proc)]
            })
            
            proc_status_chart = alt.Chart(proc_status).mark_bar().encode(
                x=alt.X('Status:N', title=None),
                y=alt.Y('Count:Q', title='Number of Admissions'),
                color=alt.Color('Status:N', 
                               scale=alt.Scale(domain=['With Procedures', 'Without Procedures'],
                                              range=['#4C78A8', '#F58518']),
                               legend=None),
                tooltip=[alt.Tooltip('Status:N'), alt.Tooltip('Count:Q', title='Admissions')]
            ).properties(height=300)
            st.altair_chart(proc_status_chart, use_container_width=True)
        else:
            st.info("No procedure data available.")


# -----------------------------------------------------------------------------
# TAB 4: OUTCOMES
# -----------------------------------------------------------------------------
with tab4:
    st.header("📈 Patient Outcomes")
    st.markdown("*Understand what happens to patients: mortality, length of stay, and outcomes by patient group.*")
    
    # -------------------------------------------------------------------------
    # 4.1 Key Outcome Metrics
    # -------------------------------------------------------------------------
    st.subheader("🔢 Key Metrics")
    
    # Calculate metrics
    hosp_mortality = admissions['hospital_expire_flag'].mean() * 100
    median_los = ((admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / 86400).median()
    icu_mortality = 0
    if len(icustays) > 0:
        icu_with_deaths = icustays.merge(
            admissions[['hadm_id', 'hospital_expire_flag']], 
            on='hadm_id'
        )
        icu_mortality = icu_with_deaths['hospital_expire_flag'].mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("In-Hospital Mortality", f"{hosp_mortality:.1f}%")
    with col2:
        st.metric("Median Hospital LOS", f"{median_los:.1f} days")
    with col3:
        st.metric("ICU Admissions", len(icustays))
    with col4:
        st.metric("ICU Mortality", f"{icu_mortality:.1f}%")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 4.2 Survivors vs Non-Survivors
    # -------------------------------------------------------------------------
    st.subheader("⚖️ Survivors vs Non-Survivors")
    st.markdown("*What's different about patients who don't survive?*")
    
    survivors = outcome_comparison_df[outcome_comparison_df['outcome'] == 'Survived']
    non_survivors = outcome_comparison_df[outcome_comparison_df['outcome'] == 'Died']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Survivors", len(survivors))
    with col2:
        st.metric("Non-Survivors", len(non_survivors))
    with col3:
        surv_age = survivors['anchor_age'].mean()
        st.metric("Avg Age (Survived)", f"{surv_age:.1f}")
    with col4:
        non_surv_age = non_survivors['anchor_age'].mean()
        st.metric("Avg Age (Died)", f"{non_surv_age:.1f}", 
                 delta=f"+{non_surv_age - surv_age:.1f} years")
    
    outcome_col1, outcome_col2 = st.columns(2)
    
    with outcome_col1:
        st.markdown("**Age Distribution by Outcome**")
        age_comparison = alt.Chart(outcome_comparison_df).mark_bar(opacity=0.7).encode(
            x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
            y=alt.Y('count()', title='Patients', stack=None),
            color=alt.Color('outcome:N', 
                           scale=alt.Scale(domain=['Survived', 'Died'], range=['#4C78A8', '#E45756']),
                           legend=alt.Legend(orient='top', title=None)),
            tooltip=[alt.Tooltip('outcome:N', title='Outcome'),
                    alt.Tooltip('anchor_age:Q', bin=alt.Bin(step=10), title='Age Range'),
                    alt.Tooltip('count()', title='Patients')]
        ).properties(height=250)
        st.altair_chart(age_comparison, use_container_width=True)
    
    with outcome_col2:
        st.markdown("**Diagnosis Burden by Outcome**")
        burden_comparison = alt.Chart(outcome_comparison_df).mark_boxplot(extent='min-max').encode(
            x=alt.X('outcome:N', title=None),
            y=alt.Y('diagnosis_count:Q', title='Number of Diagnoses'),
            color=alt.Color('outcome:N', 
                           scale=alt.Scale(domain=['Survived', 'Died'], range=['#4C78A8', '#E45756']),
                           legend=None)
        ).properties(height=250)
        st.altair_chart(burden_comparison, use_container_width=True)
    
    # Top diagnoses by outcome
    st.markdown("**Top Diagnoses by Outcome Group**")
    dx_col1, dx_col2 = st.columns(2)
    
    with dx_col1:
        st.caption("**Survivors** — Most common diagnoses")
        survivor_ids = survivors['subject_id'].tolist()
        survivor_dx = (
            dx_with_names[dx_with_names['subject_id'].isin(survivor_ids)]
            .groupby('long_title')['subject_id'].nunique()
            .reset_index().rename(columns={'subject_id': 'count'})
            .sort_values('count', ascending=False).head(8)
        )
        survivor_dx['short_title'] = survivor_dx['long_title'].apply(lambda x: smart_truncate(x, 30))
        
        survivor_chart = alt.Chart(survivor_dx).mark_bar(color='#4C78A8').encode(
            x=alt.X('count:Q', title='Patients'),
            y=alt.Y('short_title:N', title=None, sort='-x'),
            tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('count:Q', title='Patients')]
        ).properties(height=220)
        st.altair_chart(survivor_chart, use_container_width=True)
    
    with dx_col2:
        st.caption("**Non-Survivors** — Most common diagnoses")
        non_survivor_ids = non_survivors['subject_id'].tolist()
        non_survivor_dx = (
            dx_with_names[dx_with_names['subject_id'].isin(non_survivor_ids)]
            .groupby('long_title')['subject_id'].nunique()
            .reset_index().rename(columns={'subject_id': 'count'})
            .sort_values('count', ascending=False).head(8)
        )
        non_survivor_dx['short_title'] = non_survivor_dx['long_title'].apply(lambda x: smart_truncate(x, 30))
        
        non_survivor_chart = alt.Chart(non_survivor_dx).mark_bar(color='#E45756').encode(
            x=alt.X('count:Q', title='Patients'),
            y=alt.Y('short_title:N', title=None, sort='-x'),
            tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('count:Q', title='Patients')]
        ).properties(height=220)
        st.altair_chart(non_survivor_chart, use_container_width=True)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 4.3 Hospital Length of Stay
    # -------------------------------------------------------------------------
    st.subheader("🏥 Hospital Length of Stay")
    
    # Calculate LOS
    admissions_los = admissions.copy()
    admissions_los['los_days'] = (admissions_los['dischtime'] - admissions_los['admittime']).dt.total_seconds() / 86400
    
    los_stats = admissions_los['los_days'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Median LOS", f"{los_stats['50%']:.1f} days")
    with col2:
        st.metric("Mean LOS", f"{los_stats['mean']:.1f} days")
    with col3:
        st.metric("25th Percentile", f"{los_stats['25%']:.1f} days")
    with col4:
        st.metric("75th Percentile", f"{los_stats['75%']:.1f} days")
    
    # LOS histogram
    los_chart = alt.Chart(admissions_los).mark_bar(color='#B73779').encode(
        x=alt.X('los_days:Q', bin=alt.Bin(maxbins=30), title='Length of Stay (days)'),
        y=alt.Y('count()', title='Number of Admissions'),
        tooltip=[alt.Tooltip('los_days:Q', bin=alt.Bin(maxbins=30), title='LOS Range'),
                alt.Tooltip('count()', title='Admissions')]
    ).properties(height=300)
    
    st.altair_chart(los_chart, use_container_width=True)
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 4.4 Mortality by Demographics
    # -------------------------------------------------------------------------
    st.subheader("📊 Mortality by Patient Group")
    
    mort_col1, mort_col2 = st.columns(2)
    
    with mort_col1:
        st.markdown("**Mortality by Insurance Type**")
        
        ins_mortality = (
            admissions_demo.groupby('insurance')
            .agg(deaths=('hospital_expire_flag', 'sum'), n=('hospital_expire_flag', 'count'))
            .reset_index()
        )
        ins_mortality['mortality'] = ins_mortality['deaths'] / ins_mortality['n']
        ins_mortality = ins_mortality.sort_values('mortality', ascending=True)
        
        overall_mort = ins_mortality['deaths'].sum() / ins_mortality['n'].sum()
        
        ins_bars = alt.Chart(ins_mortality).mark_bar(color='#E45756').encode(
            y=alt.Y('insurance:N', title=None, sort=None),
            x=alt.X('mortality:Q', title='Mortality Rate', axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('insurance:N', title='Insurance'),
                    alt.Tooltip('mortality:Q', format='.1%', title='Mortality'),
                    alt.Tooltip('deaths:Q', title='Deaths'),
                    alt.Tooltip('n:Q', title='Total')]
        ).properties(height=200)
        
        ins_rule = alt.Chart(pd.DataFrame({'x': [overall_mort]})).mark_rule(
            strokeDash=[5, 5], color='gray'
        ).encode(x='x:Q')
        
        st.altair_chart(ins_bars + ins_rule, use_container_width=True)
        st.caption(f"Dashed line = overall mortality ({overall_mort:.1%})")
    
    with mort_col2:
        st.markdown("**Mortality by Race/Ethnicity**")
        
        race_mortality = (
            admissions_demo.groupby('race_simplified')
            .agg(deaths=('hospital_expire_flag', 'sum'), n=('hospital_expire_flag', 'count'))
            .reset_index()
        )
        race_mortality['mortality'] = race_mortality['deaths'] / race_mortality['n']
        race_mortality = race_mortality.sort_values('mortality', ascending=True)
        
        race_bars = alt.Chart(race_mortality).mark_bar(color='#E45756').encode(
            y=alt.Y('race_simplified:N', title=None, sort=None),
            x=alt.X('mortality:Q', title='Mortality Rate', axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('race_simplified:N', title='Race'),
                    alt.Tooltip('mortality:Q', format='.1%', title='Mortality'),
                    alt.Tooltip('deaths:Q', title='Deaths'),
                    alt.Tooltip('n:Q', title='Total')]
        ).properties(height=200)
        
        race_rule = alt.Chart(pd.DataFrame({'x': [overall_mort]})).mark_rule(
            strokeDash=[5, 5], color='gray'
        ).encode(x='x:Q')
        
        st.altair_chart(race_bars + race_rule, use_container_width=True)
        st.caption(f"Dashed line = overall mortality ({overall_mort:.1%})")
    
    st.divider()
    
    # -------------------------------------------------------------------------
    # 4.5 ICU Outcomes
    # -------------------------------------------------------------------------
    st.subheader("🏨 ICU Outcomes")
    
    if len(icustays) > 0:
        icu_col1, icu_col2 = st.columns(2)
        
        with icu_col1:
            st.markdown("**ICU Length of Stay by Unit**")
            
            icustays['icu_los_days'] = icustays['los']  # Already in days in MIMIC
            
            icu_los_by_unit = icustays.groupby('first_careunit')['los'].median().reset_index()
            icu_los_by_unit.columns = ['unit', 'median_los']
            icu_los_by_unit = icu_los_by_unit.sort_values('median_los', ascending=True)
            icu_los_by_unit['short_unit'] = icu_los_by_unit['unit'].apply(lambda x: smart_truncate(x, 20))
            
            icu_los_chart = alt.Chart(icu_los_by_unit).mark_bar(color='#F58518').encode(
                y=alt.Y('short_unit:N', title=None, sort=None),
                x=alt.X('median_los:Q', title='Median LOS (days)'),
                tooltip=[alt.Tooltip('unit:N', title='ICU Unit'),
                        alt.Tooltip('median_los:Q', format='.1f', title='Median LOS')]
            ).properties(height=250)
            st.altair_chart(icu_los_chart, use_container_width=True)
        
        with icu_col2:
            st.markdown("**ICU Mortality by Unit**")
            
            icu_with_outcomes = icustays.merge(
                admissions[['hadm_id', 'hospital_expire_flag']], 
                on='hadm_id'
            )
            
            icu_mort_by_unit = (
                icu_with_outcomes.groupby('first_careunit')
                .agg(deaths=('hospital_expire_flag', 'sum'), n=('hospital_expire_flag', 'count'))
                .reset_index()
            )
            icu_mort_by_unit['mortality'] = icu_mort_by_unit['deaths'] / icu_mort_by_unit['n']
            icu_mort_by_unit = icu_mort_by_unit.sort_values('mortality', ascending=True)
            icu_mort_by_unit['short_unit'] = icu_mort_by_unit['first_careunit'].apply(lambda x: smart_truncate(x, 20))
            
            overall_icu_mort = icu_mort_by_unit['deaths'].sum() / icu_mort_by_unit['n'].sum()
            
            icu_mort_chart = alt.Chart(icu_mort_by_unit).mark_bar(color='#E45756').encode(
                y=alt.Y('short_unit:N', title=None, sort=None),
                x=alt.X('mortality:Q', title='Mortality Rate', axis=alt.Axis(format='%')),
                tooltip=[alt.Tooltip('first_careunit:N', title='ICU Unit'),
                        alt.Tooltip('mortality:Q', format='.1%', title='Mortality'),
                        alt.Tooltip('deaths:Q', title='Deaths'),
                        alt.Tooltip('n:Q', title='Total')]
            ).properties(height=250)
            
            icu_rule = alt.Chart(pd.DataFrame({'x': [overall_icu_mort]})).mark_rule(
                strokeDash=[5, 5], color='gray'
            ).encode(x='x:Q')
            
            st.altair_chart(icu_mort_chart + icu_rule, use_container_width=True)
            st.caption(f"Dashed line = overall ICU mortality ({overall_icu_mort:.1%})")
    else:
        st.info("No ICU stay data available.")


# -----------------------------------------------------------------------------
# TAB 5: DEEP DIVE
# -----------------------------------------------------------------------------
with tab5:
    st.header("🔬 Diagnosis Deep Dive")
    st.markdown("*Explore a single diagnosis in detail, compare two conditions, or create contingency tables.*")
    
    # -------------------------------------------------------------------------
    # Mode Selector
    # -------------------------------------------------------------------------
    analysis_mode = st.radio(
        "Choose analysis mode:",
        ["🔍 Explore One Diagnosis", "⚖️ Compare Two Diagnoses", "📋 Contingency Tables"],
        horizontal=True,
        help="Single mode for deep exploration; Compare mode for side-by-side analysis"
    )
    
    st.divider()
    
    diagnosis_options = dx_counts['long_title'].tolist()
    
    # =========================================================================
    # SINGLE DIAGNOSIS MODE
    # =========================================================================
    if analysis_mode == "🔍 Explore One Diagnosis":
        
        st.subheader("🎯 Select a Diagnosis")
        
        # Set default based on session state (from clicking in Tab 2)
        default_index = 0
        if st.session_state.selected_diagnosis and st.session_state.selected_diagnosis in diagnosis_options:
            default_index = diagnosis_options.index(st.session_state.selected_diagnosis)
        
        selected_diagnosis = st.selectbox(
            "Choose a diagnosis to explore:",
            options=diagnosis_options,
            index=default_index,
            help="Choose a diagnosis to see detailed patient information. You can also click a bar in the Diagnoses tab!"
        )
        
        st.session_state.selected_diagnosis = selected_diagnosis
        
        # Get comprehensive stats
        stats = get_diagnosis_stats(selected_diagnosis, dx_counts, dx_with_names, patients, 
                                    admissions, icustays, diagnosis_burden)
        
        st.info(f"📌 **{selected_diagnosis}** — {stats['n_patients']} patients")
        
        st.divider()
        
        # Demographics
        st.subheader("👥 Demographics")
        
        demo_col1, demo_col2 = st.columns(2)
        
        with demo_col1:
            age_chart = alt.Chart(stats['patient_data']).mark_bar(color='#B73779').encode(
                x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
                y=alt.Y('count()', title='Number of Patients'),
                tooltip=[alt.Tooltip('anchor_age:Q', bin=alt.Bin(step=10), title='Age Range'),
                        alt.Tooltip('count()', title='Patients')]
            ).properties(title='Age Distribution', height=250)
            st.altair_chart(age_chart, use_container_width=True)
        
        with demo_col2:
            gender_counts = stats['patient_data']['gender'].value_counts().reset_index()
            gender_counts.columns = ['gender', 'count']
            
            gender_chart = alt.Chart(gender_counts).mark_bar().encode(
                x=alt.X('gender:N', title='Gender'),
                y=alt.Y('count:Q', title='Number of Patients'),
                color=alt.Color('gender:N', scale=alt.Scale(domain=['F', 'M'], range=['#E97DBB', '#7C43BD']), legend=None),
                tooltip=[alt.Tooltip('gender:N', title='Gender'), alt.Tooltip('count:Q', title='Patients')]
            ).properties(title='Gender Distribution', height=250)
            st.altair_chart(gender_chart, use_container_width=True)
        
        st.divider()
        
        # Outcomes
        st.subheader("📋 Outcomes")
        
        out_col1, out_col2, out_col3 = st.columns(3)
        with out_col1:
            delta_mort = stats['mortality'] - overall_mortality
            st.metric("Mortality Rate", f"{stats['mortality']:.1f}%", 
                     delta=f"{delta_mort:+.1f}% vs overall", delta_color="inverse")
        with out_col2:
            st.metric("Average Age", f"{stats['avg_age']:.1f} years")
        with out_col3:
            st.metric("Admissions with ICU", f"{stats['icu_rate']:.0f}%")
        
        st.divider()
        
        # Comorbidities
        st.subheader("🔀 Top Comorbidities")
        
        comorbidities = get_comorbidities(dx_with_names, stats['patient_ids'], stats['icd_code'], top_n=10)
        
        if len(comorbidities) > 0:
            comorbidities['short_title'] = comorbidities['long_title'].apply(lambda x: smart_truncate(x, 40))
            
            comorbidity_chart = alt.Chart(comorbidities).mark_bar(color='#72B7B2').encode(
                x=alt.X('patient_count:Q', title='Number of Patients'),
                y=alt.Y('short_title:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), 
                        alt.Tooltip('patient_count:Q', title='Patients')]
            ).properties(height=300)
            st.altair_chart(comorbidity_chart, use_container_width=True)
        else:
            st.info("No comorbidities found.")
        
        st.divider()
        
        # Labs & Medications
        st.subheader("🧪 Labs & Medications")
        
        lab_col1, lab_col2 = st.columns(2)
        
        with lab_col1:
            st.markdown("**Lab Value Abnormalities**")
            st.caption("Percentage of values outside reference range")
            
            lab_data, ref_ranges, top_labs = get_lab_distributions(labevents, d_labitems, stats['patient_ids'], top_n=6)
            
            if len(lab_data) > 0:
                lab_summary = ref_ranges.copy()
                lab_summary['pct_abnormal_display'] = (lab_summary['pct_abnormal'] * 100).round(0)
                lab_summary['short_label'] = lab_summary['label'].apply(lambda x: smart_truncate(x, 12))
                
                abnormal_chart = alt.Chart(lab_summary).mark_bar().encode(
                    x=alt.X('pct_abnormal_display:Q', title='% Abnormal', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('short_label:N', title=None, 
                           sort=alt.EncodingSortField('n_measurements', order='descending')),
                    color=alt.Color('pct_abnormal_display:Q', scale=alt.Scale(scheme='reds', domain=[0, 100]), legend=None),
                    tooltip=[alt.Tooltip('label:N', title='Lab'),
                            alt.Tooltip('n_measurements:Q', title='Measurements'),
                            alt.Tooltip('pct_abnormal_display:Q', title='% Abnormal')]
                ).properties(height=200)
                
                st.altair_chart(abnormal_chart, use_container_width=True)
            else:
                st.info("No lab values available for these patients.")
        
        with lab_col2:
            st.markdown("**Top Medications**")
            
            top_meds = get_top_meds_for_patients(prescriptions, stats['patient_ids']).head(6)
            
            if len(top_meds) > 0:
                top_meds['short_drug'] = top_meds['drug'].apply(lambda x: smart_truncate(x, 15))
                
                med_chart = alt.Chart(top_meds).mark_bar(color='#4C78A8').encode(
                    x=alt.X('count:Q', title='Prescriptions'),
                    y=alt.Y('short_drug:N', title=None, sort='-x'),
                    tooltip=[alt.Tooltip('drug:N', title='Medication'), 
                            alt.Tooltip('count:Q', title='Prescriptions')]
                ).properties(height=200)
                st.altair_chart(med_chart, use_container_width=True)
            else:
                st.info("No medication data available.")
    
    # =========================================================================
    # COMPARISON MODE
    # =========================================================================
    elif analysis_mode == "⚖️ Compare Two Diagnoses":
        
        st.subheader("⚖️ Compare Two Diagnoses")
        st.markdown("*Select two conditions to see how their patient populations differ.*")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            default_a = 0
            if st.session_state.selected_diagnosis and st.session_state.selected_diagnosis in diagnosis_options:
                default_a = diagnosis_options.index(st.session_state.selected_diagnosis)
            
            diagnosis_a = st.selectbox("🔵 Diagnosis A:", options=diagnosis_options, index=default_a, key="dx_a")
        
        with comp_col2:
            default_b = 1 if default_a == 0 else 0
            diagnosis_b = st.selectbox("🟠 Diagnosis B:", options=diagnosis_options, index=default_b, key="dx_b")
        
        if diagnosis_a == diagnosis_b:
            st.warning("⚠️ Please select two different diagnoses to compare.")
            st.stop()
        
        # Get stats for both
        stats_a = get_diagnosis_stats(diagnosis_a, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden)
        stats_b = get_diagnosis_stats(diagnosis_b, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden)
        
        color_a = '#4C78A8'  # Blue
        color_b = '#F58518'  # Orange
        
        st.divider()
        
        # Key Metrics Comparison
        st.subheader("📊 Key Metrics Comparison")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"**🔵 {smart_truncate(diagnosis_a, 40)}**")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Patients", stats_a['n_patients'])
                st.metric("Avg Age", f"{stats_a['avg_age']:.1f} yrs")
            with m2:
                st.metric("Mortality", f"{stats_a['mortality']:.1f}%")
                st.metric("ICU Rate", f"{stats_a['icu_rate']:.0f}%")
            with m3:
                st.metric("Median Dx", f"{stats_a['median_burden']:.0f}")
                st.metric("% Female", f"{stats_a['pct_female']:.0f}%")
        
        with metric_col2:
            st.markdown(f"**🟠 {smart_truncate(diagnosis_b, 40)}**")
            delta_patients = stats_b['n_patients'] - stats_a['n_patients']
            delta_mort = stats_b['mortality'] - stats_a['mortality']
            delta_age = stats_b['avg_age'] - stats_a['avg_age']
            delta_icu = stats_b['icu_rate'] - stats_a['icu_rate']
            delta_burden = stats_b['median_burden'] - stats_a['median_burden']
            delta_female = stats_b['pct_female'] - stats_a['pct_female']
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Patients", stats_b['n_patients'], delta=f"{delta_patients:+d}")
                st.metric("Avg Age", f"{stats_b['avg_age']:.1f} yrs", delta=f"{delta_age:+.1f}")
            with m2:
                st.metric("Mortality", f"{stats_b['mortality']:.1f}%", delta=f"{delta_mort:+.1f}%", delta_color="inverse")
                st.metric("ICU Rate", f"{stats_b['icu_rate']:.0f}%", delta=f"{delta_icu:+.0f}%", delta_color="inverse")
            with m3:
                st.metric("Median Dx", f"{stats_b['median_burden']:.0f}", delta=f"{delta_burden:+.0f}")
                st.metric("% Female", f"{stats_b['pct_female']:.0f}%", delta=f"{delta_female:+.0f}%")
        
        st.divider()
        
        # Age Distribution Comparison
        st.subheader("👥 Age Distribution")
        
        patients_a = stats_a['patient_data'].copy()
        patients_a['group'] = smart_truncate(diagnosis_a, 25)
        patients_b = stats_b['patient_data'].copy()
        patients_b['group'] = smart_truncate(diagnosis_b, 25)
        combined_patients = pd.concat([patients_a, patients_b])
        
        age_comparison = alt.Chart(combined_patients).mark_bar(opacity=0.6).encode(
            x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
            y=alt.Y('count()', title='Patients', stack=None),
            color=alt.Color('group:N', 
                           scale=alt.Scale(domain=[smart_truncate(diagnosis_a, 25), smart_truncate(diagnosis_b, 25)], 
                                          range=[color_a, color_b]),
                           legend=alt.Legend(orient='top', title=None)),
            tooltip=[alt.Tooltip('group:N', title='Diagnosis'),
                    alt.Tooltip('anchor_age:Q', bin=alt.Bin(step=10), title='Age Range'),
                    alt.Tooltip('count()', title='Patients')]
        ).properties(height=300)
        
        st.altair_chart(age_comparison, use_container_width=True)
        
        st.divider()
        
        # Comorbidities Comparison
        st.subheader("🔀 Top Comorbidities")
        
        comorb_col1, comorb_col2 = st.columns(2)
        
        with comorb_col1:
            st.caption(f"**🔵 {smart_truncate(diagnosis_a, 35)}**")
            comorb_a = get_comorbidities_for_comparison(dx_with_names, stats_a['patient_ids'], 
                                                        [stats_a['icd_code'], stats_b['icd_code']], top_n=8)
            if len(comorb_a) > 0:
                comorb_a['short_title'] = comorb_a['long_title'].apply(lambda x: smart_truncate(x, 30))
                chart_a = alt.Chart(comorb_a).mark_bar(color=color_a).encode(
                    x=alt.X('patient_count:Q', title='Patients'),
                    y=alt.Y('short_title:N', title=None, sort='-x'),
                    tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), 
                            alt.Tooltip('patient_count:Q', title='Patients')]
                ).properties(height=250)
                st.altair_chart(chart_a, use_container_width=True)
        
        with comorb_col2:
            st.caption(f"**🟠 {smart_truncate(diagnosis_b, 35)}**")
            comorb_b = get_comorbidities_for_comparison(dx_with_names, stats_b['patient_ids'], 
                                                        [stats_a['icd_code'], stats_b['icd_code']], top_n=8)
            if len(comorb_b) > 0:
                comorb_b['short_title'] = comorb_b['long_title'].apply(lambda x: smart_truncate(x, 30))
                chart_b = alt.Chart(comorb_b).mark_bar(color=color_b).encode(
                    x=alt.X('patient_count:Q', title='Patients'),
                    y=alt.Y('short_title:N', title=None, sort='-x'),
                    tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), 
                            alt.Tooltip('patient_count:Q', title='Patients')]
                ).properties(height=250)
                st.altair_chart(chart_b, use_container_width=True)
        
        # Shared comorbidities
        if len(comorb_a) > 0 and len(comorb_b) > 0:
            shared = set(comorb_a['icd_code']) & set(comorb_b['icd_code'])
            if shared:
                shared_names = comorb_a[comorb_a['icd_code'].isin(shared)]['long_title'].tolist()[:3]
                st.info(f"🔗 **Shared comorbidities:** {', '.join([smart_truncate(n, 30) for n in shared_names])}")
        
        st.divider()
        
        # Patient Overlap
        st.subheader("🔄 Patient Overlap")
        
        set_a = set(stats_a['patient_ids'])
        set_b = set(stats_b['patient_ids'])
        overlap = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        
        overlap_col1, overlap_col2, overlap_col3 = st.columns(3)
        with overlap_col1:
            st.metric(f"Only 🔵", len(only_a))
        with overlap_col2:
            st.metric("Have Both", len(overlap))
        with overlap_col3:
            st.metric(f"Only 🟠", len(only_b))
        
        overlap_data = pd.DataFrame({
            'category': ['Only 🔵', 'Both', 'Only 🟠'],
            'count': [len(only_a), len(overlap), len(only_b)]
        })
        
        overlap_chart = alt.Chart(overlap_data).mark_bar().encode(
            x=alt.X('category:N', title=None, sort=['Only 🔵', 'Both', 'Only 🟠']),
            y=alt.Y('count:Q', title='Patients'),
            color=alt.Color('category:N', 
                           scale=alt.Scale(domain=['Only 🔵', 'Both', 'Only 🟠'], 
                                          range=[color_a, '#72B7B2', color_b]), 
                           legend=None),
            tooltip=[alt.Tooltip('category:N', title='Group'), alt.Tooltip('count:Q', title='Patients')]
        ).properties(height=200)
        
        st.altair_chart(overlap_chart, use_container_width=True)
        
        if len(overlap) > 0:
            pct_overlap_a = len(overlap) / len(set_a) * 100
            pct_overlap_b = len(overlap) / len(set_b) * 100
            st.caption(f"💡 {len(overlap)} patients have both diagnoses ({pct_overlap_a:.0f}% of 🔵, {pct_overlap_b:.0f}% of 🟠)")
    
    # =========================================================================
    # CONTINGENCY TABLES MODE
    # =========================================================================
    else:  # Contingency Tables
        
        st.subheader("📋 Contingency Tables")
        st.markdown("*Cross-tabulate categorical variables to explore relationships.*")
        
        # Available fields for contingency tables
        field_labels = {
            "race_simplified": "Race/Ethnicity",
            "admission_loc_simple": "Admission Location",
            "admission_type_simple": "Admission Type",
            "discharge_loc_simple": "Discharge Location",
            "insurance": "Insurance",
            "marital_status": "Marital Status"
        }
        label_to_field = {v: k for k, v in field_labels.items()}
        
        selected_labels = st.multiselect(
            "Choose up to 3 categorical variables (Row, Column, Facet)",
            options=list(field_labels.values()),
            default=["Race/Ethnicity", "Admission Type"],
            max_selections=3
        )
        
        if not selected_labels:
            st.info("Select 1–3 variables to see a contingency table.")
        else:
            selected_fields = [label_to_field[label] for label in selected_labels]
            row_field = selected_fields[0]
            col_field = selected_fields[1] if len(selected_fields) >= 2 else None
            facet_field = selected_fields[2] if len(selected_fields) == 3 else None
            
            def show_crosstab(df, row_f, col_f=None, facet_val=None):
                """Display a crosstab table."""
                if col_f:
                    table = pd.crosstab(df[row_f], df[col_f], margins=True)
                else:
                    table = df[row_f].value_counts().rename("Count").to_frame()
                
                title_parts = []
                if facet_val:
                    title_parts.append(f"{field_labels.get(facet_field, facet_field)}: {facet_val}")
                
                if title_parts:
                    st.markdown(f"**{', '.join(title_parts)}** (n={len(df)})")
                
                st.dataframe(table, use_container_width=True)
            
            if facet_field:
                for facet_val in sorted(admissions_demo[facet_field].dropna().unique()):
                    subset = admissions_demo[admissions_demo[facet_field] == facet_val]
                    show_crosstab(subset, row_field, col_field, facet_val)
                    st.markdown("---")
            else:
                show_crosstab(admissions_demo, row_field, col_field)


# =============================================================================
# SECTION 8: DATA PREVIEW (Development Tool)
# =============================================================================

st.divider()

with st.expander("🔍 Preview Raw Data (click to expand)"):
    table_choice = st.selectbox(
        "Select a table to preview:",
        ["patients", "admissions", "diagnoses_icd", "d_icd_diagnoses", 
         "labevents", "d_labitems", "prescriptions", "pharmacy",
         "icustays", "procedures_icd", "d_icd_procedures", "dx_counts (processed)"]
    )
    
    table_map = {
        "patients": patients,
        "admissions": admissions,
        "diagnoses_icd": diagnoses,
        "d_icd_diagnoses": d_icd_diagnoses,
        "labevents": labevents,
        "d_labitems": d_labitems,
        "prescriptions": prescriptions,
        "pharmacy": pharmacy,
        "icustays": icustays,
        "procedures_icd": procedures,
        "d_icd_procedures": d_icd_procedures,
        "dx_counts (processed)": dx_counts
    }
    
    st.dataframe(table_map[table_choice].head(20), use_container_width=True)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("MIMIC-Explorer | Harvard Medical School | Data Visualization for Biomedical Applications | 2025")
