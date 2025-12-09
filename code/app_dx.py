"""
MIMIC-Explorer: Diagnosis Explorer
A Streamlit dashboard for exploring diagnosis patterns in MIMIC-IV

Structure:
    PART 1: Dataset Overview — Understand the landscape
    PART 2: Diagnosis Deep Dive — Explore or Compare specific conditions

To run this app:
    cd code/
    streamlit run app.py
"""

# =============================================================================
# SECTION 1: IMPORTS & PAGE CONFIG
# =============================================================================
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="MIMIC-Explorer",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for selected diagnosis
if 'selected_diagnosis' not in st.session_state:
    st.session_state.selected_diagnosis = None

# =============================================================================
# SECTION 2: DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data
def load_data():
    """Load all required MIMIC-IV tables."""
    data_path = "../data/"
    
    patients = pd.read_csv(data_path + "patients.csv")
    admissions = pd.read_csv(data_path + "admissions.csv")
    diagnoses = pd.read_csv(data_path + "diagnoses_icd.csv")
    d_icd = pd.read_csv(data_path + "d_icd_diagnoses.csv")
    
    labevents = pd.read_csv(data_path + "labevents.csv")
    d_labitems = pd.read_csv(data_path + "d_labitems.csv")
    prescriptions = pd.read_csv(data_path + "prescriptions.csv")
    icustays = pd.read_csv(data_path + "icustays.csv")
    
    return patients, admissions, diagnoses, d_icd, labevents, d_labitems, prescriptions, icustays


# =============================================================================
# SECTION 3: DATA PREPARATION FUNCTIONS
# =============================================================================
@st.cache_data
def prepare_diagnosis_data(diagnoses, d_icd):
    """Merge diagnoses with descriptions and count occurrences."""
    dx_with_names = diagnoses.merge(
        d_icd,
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
def get_diagnosis_patient_data(diagnoses, d_icd, patients):
    """Create merged dataset linking diagnoses to patient demographics."""
    dx_with_names = diagnoses.merge(
        d_icd,
        on=['icd_code', 'icd_version'],
        how='left'
    )
    
    dx_with_patients = dx_with_names.merge(
        patients[['subject_id', 'gender', 'anchor_age']],
        on='subject_id',
        how='left'
    )
    
    return dx_with_patients


@st.cache_data
def prepare_cooccurrence_matrix(dx_with_names, dx_counts, top_n=15):
    """Build co-occurrence matrix for heatmap visualization."""
    top_diagnoses = dx_counts.head(top_n)['icd_code'].tolist()
    dx_filtered = dx_with_names[dx_with_names['icd_code'].isin(top_diagnoses)]
    patient_dx = dx_filtered.groupby('subject_id')['icd_code'].apply(set).reset_index()
    
    from itertools import combinations
    cooccurrence = {}
    
    for _, row in patient_dx.iterrows():
        dx_list = list(row['icd_code'])
        for dx in dx_list:
            cooccurrence[(dx, dx)] = cooccurrence.get((dx, dx), 0) + 1
        for dx1, dx2 in combinations(dx_list, 2):
            cooccurrence[(dx1, dx2)] = cooccurrence.get((dx1, dx2), 0) + 1
            cooccurrence[(dx2, dx1)] = cooccurrence.get((dx2, dx1), 0) + 1
    
    records = []
    for (dx1, dx2), count in cooccurrence.items():
        records.append({'icd_code_1': dx1, 'icd_code_2': dx2, 'count': count})
    
    cooccurrence_df = pd.DataFrame(records)
    
    from itertools import product
    all_pairs = pd.DataFrame(
        list(product(top_diagnoses, top_diagnoses)),
        columns=['icd_code_1', 'icd_code_2']
    )
    cooccurrence_df = all_pairs.merge(cooccurrence_df, on=['icd_code_1', 'icd_code_2'], how='left')
    cooccurrence_df['count'] = cooccurrence_df['count'].fillna(0).astype(int)
    
    code_to_name = dict(zip(dx_counts['icd_code'], dx_counts['long_title']))
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
def get_top_labs_for_patients(labevents, d_labitems, patient_ids, top_n=10):
    """Get the most common lab tests for a set of patients."""
    labs_filtered = labevents[labevents['subject_id'].isin(patient_ids)]
    lab_counts = (
        labs_filtered
        .merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
        .groupby('label')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)
    )
    return lab_counts


@st.cache_data
def get_top_meds_for_patients(prescriptions, patient_ids, top_n=10):
    """Get the most common medications for a set of patients."""
    meds_filtered = prescriptions[prescriptions['subject_id'].isin(patient_ids)]
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
def get_comorbidities(dx_with_names, patient_ids, exclude_icd_code, top_n=10):
    """Get the most common OTHER diagnoses for a set of patients."""
    comorbidities = (
        dx_with_names[
            (dx_with_names['subject_id'].isin(patient_ids)) & 
            (dx_with_names['icd_code'] != exclude_icd_code)
        ]
        .groupby(['icd_code', 'long_title'])['subject_id']
        .nunique()
        .reset_index(name='patient_count')
        .sort_values('patient_count', ascending=False)
        .head(top_n)
    )
    return comorbidities


@st.cache_data
def get_comorbidities_for_comparison(dx_with_names, patient_ids, exclude_icd_codes, top_n=10):
    """Get comorbidities excluding multiple diagnoses (for comparison mode)."""
    comorbidities = (
        dx_with_names[
            (dx_with_names['subject_id'].isin(patient_ids)) & 
            (~dx_with_names['icd_code'].isin(exclude_icd_codes))
        ]
        .groupby(['icd_code', 'long_title'])['subject_id']
        .nunique()
        .reset_index(name='patient_count')
        .sort_values('patient_count', ascending=False)
        .head(top_n)
    )
    return comorbidities


@st.cache_data
def calculate_diagnosis_burden(diagnoses):
    """Calculate how many unique diagnoses each patient has."""
    burden = (
        diagnoses
        .groupby('subject_id')['icd_code']
        .nunique()
        .reset_index()
        .rename(columns={'icd_code': 'diagnosis_count'})
    )
    return burden


@st.cache_data
def get_age_by_top_diagnoses(dx_with_patients, dx_counts, top_n=6):
    """Get age data for patients with each of the top N diagnoses."""
    top_dx_codes = dx_counts.head(top_n)['icd_code'].tolist()
    
    age_by_dx = (
        dx_with_patients[dx_with_patients['icd_code'].isin(top_dx_codes)]
        [['subject_id', 'icd_code', 'long_title', 'anchor_age']]
        .drop_duplicates()
    )
    
    age_by_dx['short_title'] = age_by_dx['long_title'].apply(
        lambda x: x[:25] + '...' if len(str(x)) > 25 else x
    )
    
    dx_patient_counts = age_by_dx.groupby('icd_code')['subject_id'].nunique().reset_index()
    dx_patient_counts.columns = ['icd_code', 'dx_patient_count']
    age_by_dx = age_by_dx.merge(dx_patient_counts, on='icd_code')
    
    return age_by_dx


@st.cache_data
def get_outcome_comparison_data(patients, admissions, diagnoses, dx_counts, diagnosis_burden):
    """Prepare comparison data for survivors vs non-survivors."""
    patient_outcomes = admissions.groupby('subject_id')['hospital_expire_flag'].max().reset_index()
    patient_outcomes.columns = ['subject_id', 'died']
    
    comparison_df = patients.merge(patient_outcomes, on='subject_id', how='left')
    comparison_df['died'] = comparison_df['died'].fillna(0).astype(int)
    comparison_df['outcome'] = comparison_df['died'].map({0: 'Survived', 1: 'Died'})
    comparison_df = comparison_df.merge(diagnosis_burden, on='subject_id', how='left')
    
    return comparison_df


@st.cache_data
def build_patient_flow_data(admissions, icustays):
    """Build data for patient flow Sankey diagram."""
    flow_df = admissions[['hadm_id', 'subject_id', 'admission_type', 'hospital_expire_flag']].copy()
    
    icu_hadm_ids = icustays['hadm_id'].unique()
    flow_df['icu_status'] = flow_df['hadm_id'].isin(icu_hadm_ids).map({True: 'ICU Stay', False: 'No ICU'})
    flow_df['outcome'] = flow_df['hospital_expire_flag'].map({0: 'Survived', 1: 'Died'})
    
    admission_counts = flow_df['admission_type'].value_counts()
    top_admissions = admission_counts[admission_counts >= 5].index.tolist()
    flow_df['admission_group'] = flow_df['admission_type'].apply(
        lambda x: x if x in top_admissions else 'Other'
    )
    
    flow1 = flow_df.groupby(['admission_group', 'icu_status']).size().reset_index(name='count')
    flow1.columns = ['source', 'target', 'value']
    
    flow2 = flow_df.groupby(['icu_status', 'outcome']).size().reset_index(name='count')
    flow2.columns = ['source', 'target', 'value']
    
    all_flows = pd.concat([flow1, flow2], ignore_index=True)
    all_nodes = list(pd.concat([all_flows['source'], all_flows['target']]).unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    all_flows['source_idx'] = all_flows['source'].map(node_indices)
    all_flows['target_idx'] = all_flows['target'].map(node_indices)
    
    return all_flows, all_nodes, node_indices


@st.cache_data
def get_lab_distributions(labevents, d_labitems, patient_ids, top_n=6):
    """Get lab value distributions for a set of patients."""
    labs_filtered = labevents[
        (labevents['subject_id'].isin(patient_ids)) & 
        (labevents['valuenum'].notna())
    ].copy()
    
    labs_filtered = labs_filtered.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
    
    top_labs = (
        labs_filtered.groupby('label')
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(top_n)['label'].tolist()
    )
    
    lab_data = labs_filtered[labs_filtered['label'].isin(top_labs)].copy()
    
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


def smart_truncate(text, max_length=40):
    """Truncate text at word boundary, not mid-word."""
    text = str(text)
    if len(text) <= max_length:
        return text
    truncated = text[:max_length].rsplit(' ', 1)[0]
    return truncated + '...'


def get_diagnosis_stats(dx_name, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden):
    """Get all stats for a single diagnosis - used in comparison mode."""
    icd_code = dx_counts[dx_counts['long_title'] == dx_name]['icd_code'].values[0]
    patient_ids = dx_with_names[dx_with_names['icd_code'] == icd_code]['subject_id'].unique()
    patient_data = patients[patients['subject_id'].isin(patient_ids)]
    
    # Mortality
    dx_admissions = admissions[admissions['subject_id'].isin(patient_ids)]
    mortality = dx_admissions['hospital_expire_flag'].mean() * 100 if len(dx_admissions) > 0 else 0
    
    # ICU rate (admission-based)
    hadm_ids = dx_with_names[dx_with_names['icd_code'] == icd_code]['hadm_id'].unique()
    icu_hadm_ids = set(icustays['hadm_id'].unique())
    icu_rate = len(set(hadm_ids) & icu_hadm_ids) / len(hadm_ids) * 100 if len(hadm_ids) > 0 else 0
    
    # Demographics
    avg_age = patient_data['anchor_age'].mean() if len(patient_data) > 0 else 0
    pct_female = (patient_data['gender'] == 'F').mean() * 100 if len(patient_data) > 0 else 0
    
    # Burden
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


# =============================================================================
# SECTION 4: LOAD DATA
# =============================================================================
patients, admissions, diagnoses, d_icd, labevents, d_labitems, prescriptions, icustays = load_data()
dx_with_names, dx_counts = prepare_diagnosis_data(diagnoses, d_icd)
dx_with_patients = get_diagnosis_patient_data(diagnoses, d_icd, patients)
diagnosis_burden = calculate_diagnosis_burden(diagnoses)
cooccurrence_df, top_dx_codes = prepare_cooccurrence_matrix(dx_with_names, dx_counts, top_n=10)
age_by_top_dx = get_age_by_top_diagnoses(dx_with_patients, dx_counts, top_n=6)
outcome_comparison_df = get_outcome_comparison_data(patients, admissions, diagnoses, dx_counts, diagnosis_burden)

# Calculate overall stats for comparison
overall_mortality = admissions['hospital_expire_flag'].mean() * 100
overall_avg_age = patients['anchor_age'].mean()
overall_median_burden = diagnosis_burden['diagnosis_count'].median()


# =============================================================================
# SECTION 5: SIDEBAR (Reserved for future use / other team pages)
# =============================================================================
# Keeping sidebar empty for now — controls are in main page near their charts


# =============================================================================
# SECTION 6: HEADER
# =============================================================================
st.title("🏥 MIMIC-Explorer")
st.markdown("### Diagnosis Explorer")
st.markdown("An interactive dashboard for exploring diagnosis patterns in MIMIC-IV")

st.divider()


# #############################################################################
#                                                                             #
#                     PART 1: DATASET OVERVIEW                                #
#                     Understand the Landscape                                #
#                                                                             #
# #############################################################################

st.markdown("## 🗺️ Part 1: Dataset Overview")
st.markdown("*Get oriented with the data before diving into specific conditions.*")

st.divider()


# =============================================================================
# 1.1: DATA OVERVIEW (Metrics Row)
# =============================================================================
st.header("📊 At a Glance")
st.markdown("*A quick snapshot of the dataset's scope and scale.*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Patients", len(patients))
    
with col2:
    st.metric("Admissions", len(admissions))
    
with col3:
    st.metric("Diagnosis Records", f"{len(diagnoses):,}")
    
with col4:
    n_unique_dx = diagnoses['icd_code'].nunique()
    st.metric("Unique Diagnoses", n_unique_dx)

st.divider()


# =============================================================================
# 1.2: TOP DIAGNOSES BAR CHART
# =============================================================================
st.header("🏆 Most Common Diagnoses")
st.markdown("*What conditions appear most frequently in this patient population?*")

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

if chart_event and chart_event.selection and 'param_1' in chart_event.selection:
    selected_points = chart_event.selection['param_1']
    if selected_points and len(selected_points) > 0:
        clicked_diagnosis = selected_points[0].get('long_title')
        if clicked_diagnosis:
            st.session_state.selected_diagnosis = clicked_diagnosis

st.caption("💡 Click any bar to select a diagnosis for deep-dive analysis in Part 2 below.")

# Small multiples and outcome comparison
st.subheader("Patterns Across Top Diagnoses")
st.markdown("*How do age and outcomes vary across the most common conditions?*")

col1, col2 = st.columns([3, 2])

with col1:
    st.caption("**Age Distribution**")
    small_multiples = alt.Chart(age_by_top_dx).mark_bar(color='#B73779').encode(
        x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
        y=alt.Y('count()', title='Patients'),
        tooltip=[
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
    st.caption("**Mortality & ICU Rates**")
    
    top_6_codes = dx_counts.head(6)['icd_code'].tolist()
    
    outcome_data = []
    for code in top_6_codes:
        dx_name = dx_counts[dx_counts['icd_code'] == code]['long_title'].values[0]
        short_name = dx_name[:20] + '...' if len(dx_name) > 20 else dx_name
        
        pts = dx_with_names[dx_with_names['icd_code'] == code]['subject_id'].unique()
        dx_admissions = admissions[admissions['subject_id'].isin(pts)]
        mortality = dx_admissions['hospital_expire_flag'].mean() * 100
        
        dx_admissions_ids = dx_with_names[dx_with_names['icd_code'] == code]['hadm_id'].unique()
        icu_hadm_ids = icustays['hadm_id'].unique()
        dx_icu_admissions = set(dx_admissions_ids) & set(icu_hadm_ids)
        icu_rate = len(dx_icu_admissions) / len(dx_admissions_ids) * 100 if len(dx_admissions_ids) > 0 else 0
        
        outcome_data.append({'diagnosis': short_name, 'Mortality %': mortality, 'ICU %': icu_rate})
    
    outcome_df = pd.DataFrame(outcome_data)
    dx_order = outcome_df['diagnosis'].tolist()
    
    mortality_chart = alt.Chart(outcome_df).mark_bar(color='#E45756').encode(
        y=alt.Y('diagnosis:N', title=None, sort=dx_order),
        x=alt.X('Mortality %:Q', title='Mortality %'),
        tooltip=[alt.Tooltip('diagnosis:N', title='Diagnosis'), alt.Tooltip('Mortality %:Q', format='.1f')]
    ).properties(height=150, title='Mortality Rate')
    
    icu_chart = alt.Chart(outcome_df).mark_bar(color='#F58518').encode(
        y=alt.Y('diagnosis:N', title=None, sort=dx_order),
        x=alt.X('ICU %:Q', title='ICU %'),
        tooltip=[alt.Tooltip('diagnosis:N', title='Diagnosis'), alt.Tooltip('ICU %:Q', format='.1f')]
    ).properties(height=150, title='ICU Admission Rate')
    
    outcome_chart = alt.vconcat(mortality_chart, icu_chart).resolve_scale(x='independent')
    st.altair_chart(outcome_chart, use_container_width=True)

st.divider()


# =============================================================================
# 1.3: PATIENT FLOW SANKEY
# =============================================================================
st.header("🌊 Patient Flow")
st.markdown("*How do patients move through the hospital system? Follow the flow from admission to outcome.*")

flow_data, flow_nodes, node_indices = build_patient_flow_data(admissions, icustays)

node_colors = []
for node in flow_nodes:
    if node == 'Died':
        node_colors.append('#E45756')
    elif node == 'Survived':
        node_colors.append('#4C78A8')
    elif node == 'ICU Stay':
        node_colors.append('#F58518')
    elif node == 'No ICU':
        node_colors.append('#72B7B2')
    else:
        node_colors.append('#B73779')

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

fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5),
              label=flow_nodes, color=node_colors),
    link=dict(source=flow_data['source_idx'].tolist(),
              target=flow_data['target_idx'].tolist(),
              value=flow_data['value'].tolist(),
              color=link_colors)
)])

fig.update_layout(
    font=dict(size=12, color='white'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)
st.caption("💡 Width of each flow represents number of patients. Hover for details. Red flows lead to mortality.")

# =============================================================================
# 1.3b: EXPLORE A PATIENT PATHWAY
# =============================================================================
st.subheader("🔎 Explore a Patient Pathway")
st.markdown("*Select a path through the hospital to see who those patients are.*")

# Build the pathway filter data
pathway_df = admissions[['hadm_id', 'subject_id', 'admission_type', 'hospital_expire_flag']].copy()
icu_hadm_set = set(icustays['hadm_id'].unique())
pathway_df['icu_status'] = pathway_df['hadm_id'].apply(lambda x: 'ICU Stay' if x in icu_hadm_set else 'No ICU')
pathway_df['outcome'] = pathway_df['hospital_expire_flag'].map({0: 'Survived', 1: 'Died'})

# Get unique values for dropdowns
admission_types = ['All'] + sorted(pathway_df['admission_type'].unique().tolist())
icu_options = ['All', 'ICU Stay', 'No ICU']
outcome_options = ['All', 'Survived', 'Died']

# Create filter dropdowns
pcol1, pcol2, pcol3 = st.columns(3)

with pcol1:
    selected_admission = st.selectbox("Admission Type:", admission_types, key="pathway_admission")
with pcol2:
    selected_icu = st.selectbox("ICU Status:", icu_options, key="pathway_icu")
with pcol3:
    selected_outcome = st.selectbox("Outcome:", outcome_options, key="pathway_outcome")

# Filter based on selections
filtered_pathway = pathway_df.copy()
if selected_admission != 'All':
    filtered_pathway = filtered_pathway[filtered_pathway['admission_type'] == selected_admission]
if selected_icu != 'All':
    filtered_pathway = filtered_pathway[filtered_pathway['icu_status'] == selected_icu]
if selected_outcome != 'All':
    filtered_pathway = filtered_pathway[filtered_pathway['outcome'] == selected_outcome]

pathway_patient_ids = filtered_pathway['subject_id'].unique()
pathway_patients = patients[patients['subject_id'].isin(pathway_patient_ids)]

# Show count
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
        st.metric("Median Dx", f"{med_burden:.0f}")
    
    # Top diagnoses and medications side by side
    pw_col1, pw_col2 = st.columns(2)
    
    with pw_col1:
        st.markdown("**Top Diagnoses**")
        pathway_dx = (
            dx_with_names[dx_with_names['subject_id'].isin(pathway_patient_ids)]
            .groupby('long_title')['subject_id'].nunique()
            .reset_index()
            .rename(columns={'subject_id': 'count'})
            .sort_values('count', ascending=False)
            .head(8)
        )
        pathway_dx['short_title'] = pathway_dx['long_title'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)
        
        if len(pathway_dx) > 0:
            dx_chart = alt.Chart(pathway_dx).mark_bar(color='#72B7B2').encode(
                x=alt.X('count:Q', title='Patients'),
                y=alt.Y('short_title:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('count:Q', title='Patients')]
            ).properties(height=220)
            st.altair_chart(dx_chart, use_container_width=True)
    
    with pw_col2:
        st.markdown("**Top Medications**")
        pathway_meds = get_top_meds_for_patients(prescriptions, pathway_patient_ids).head(8)
        
        if len(pathway_meds) > 0:
            pathway_meds['short_drug'] = pathway_meds['drug'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
            med_chart = alt.Chart(pathway_meds).mark_bar(color='#72B7B2').encode(
                x=alt.X('count:Q', title='Prescriptions'),
                y=alt.Y('short_drug:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('drug:N', title='Medication'), alt.Tooltip('count:Q', title='Prescriptions')]
            ).properties(height=220)
            st.altair_chart(med_chart, use_container_width=True)
        else:
            st.info("No medication data for this pathway.")
else:
    st.warning("No patients match this pathway combination.")

st.divider()


# =============================================================================
# 1.4: DIAGNOSIS CO-OCCURRENCE HEATMAP
# =============================================================================
st.header("🔗 Diagnosis Co-occurrence")
st.markdown("*Which diagnoses frequently appear together? Brighter cells indicate stronger co-occurrence.*")

sort_order = dx_counts[dx_counts['icd_code'].isin(top_dx_codes)]['long_title'].tolist()
short_sort_order = [x[:20] + '...' if len(x) > 20 else x for x in sort_order]

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
).properties(width=600, height=600)

st.altair_chart(heatmap, use_container_width=True)
st.caption("💡 Brighter cells indicate diagnoses that frequently co-occur. The diagonal shows how many patients have each diagnosis.")

st.divider()


# =============================================================================
# 1.5: DIAGNOSIS BURDEN DISTRIBUTION
# =============================================================================
st.header("📈 Patient Complexity")
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

burden_hist = alt.Chart(diagnosis_burden).mark_bar().encode(
    x=alt.X('diagnosis_count:Q', bin=alt.Bin(step=5), title='Number of Diagnoses per Patient'),
    y=alt.Y('count()', title='Number of Patients'),
    color=alt.value('#B73779'),
    tooltip=[
        alt.Tooltip('diagnosis_count:Q', bin=alt.Bin(step=5), title='Diagnosis Range'),
        alt.Tooltip('count()', title='Patients')
    ]
).properties(height=300)

median_line = alt.Chart(pd.DataFrame({'median': [median_burden]})).mark_rule(
    color='red', strokeDash=[5, 5], strokeWidth=2
).encode(x='median:Q')

burden_chart = burden_hist + median_line
st.altair_chart(burden_chart, use_container_width=True)
st.caption(f"📊 Red dashed line = median ({median_burden:.0f} diagnoses). Note the right skew — some patients have 100+ diagnoses.")

st.divider()


# =============================================================================
# 1.6: SURVIVORS VS NON-SURVIVORS
# =============================================================================
st.header("⚖️ Survivors vs Non-Survivors")
st.markdown("*What's different about patients who don't survive? A side-by-side comparison.*")

survivors = outcome_comparison_df[outcome_comparison_df['outcome'] == 'Survived']
non_survivors = outcome_comparison_df[outcome_comparison_df['outcome'] == 'Died']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Survivors", len(survivors))
with col2:
    st.metric("Non-Survivors", len(non_survivors))
with col3:
    surv_age = survivors['anchor_age'].mean()
    non_surv_age = non_survivors['anchor_age'].mean()
    st.metric("Avg Age (Survived)", f"{surv_age:.1f}")
with col4:
    st.metric("Avg Age (Died)", f"{non_surv_age:.1f}", delta=f"+{non_surv_age - surv_age:.1f} years")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution")
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

with col2:
    st.subheader("Diagnosis Burden")
    burden_comparison = alt.Chart(outcome_comparison_df).mark_boxplot(extent='min-max').encode(
        x=alt.X('outcome:N', title=None),
        y=alt.Y('diagnosis_count:Q', title='Number of Diagnoses'),
        color=alt.Color('outcome:N', 
                        scale=alt.Scale(domain=['Survived', 'Died'], range=['#4C78A8', '#E45756']),
                        legend=None)
    ).properties(height=250)
    st.altair_chart(burden_comparison, use_container_width=True)

st.subheader("Top Diagnoses by Outcome")

col1, col2 = st.columns(2)

with col1:
    st.caption("**Survivors** — Most common diagnoses")
    survivor_ids = survivors['subject_id'].tolist()
    survivor_dx = (
        dx_with_names[dx_with_names['subject_id'].isin(survivor_ids)]
        .groupby('long_title')['subject_id'].nunique()
        .reset_index().rename(columns={'subject_id': 'count'})
        .sort_values('count', ascending=False).head(8)
    )
    survivor_dx['short_title'] = survivor_dx['long_title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    survivor_chart = alt.Chart(survivor_dx).mark_bar(color='#4C78A8').encode(
        x=alt.X('count:Q', title='Patients'),
        y=alt.Y('short_title:N', title=None, sort='-x'),
        tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('count:Q', title='Patients')]
    ).properties(height=220)
    st.altair_chart(survivor_chart, use_container_width=True)

with col2:
    st.caption("**Non-Survivors** — Most common diagnoses")
    non_survivor_ids = non_survivors['subject_id'].tolist()
    non_survivor_dx = (
        dx_with_names[dx_with_names['subject_id'].isin(non_survivor_ids)]
        .groupby('long_title')['subject_id'].nunique()
        .reset_index().rename(columns={'subject_id': 'count'})
        .sort_values('count', ascending=False).head(8)
    )
    non_survivor_dx['short_title'] = non_survivor_dx['long_title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    non_survivor_chart = alt.Chart(non_survivor_dx).mark_bar(color='#E45756').encode(
        x=alt.X('count:Q', title='Patients'),
        y=alt.Y('short_title:N', title=None, sort='-x'),
        tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('count:Q', title='Patients')]
    ).properties(height=220)
    st.altair_chart(non_survivor_chart, use_container_width=True)

st.caption("💡 Compare patterns between groups. Look for diagnoses that appear more frequently in non-survivors.")

st.divider()


# #############################################################################
#                                                                             #
#                     PART 2: DIAGNOSIS DEEP DIVE                             #
#                     Explore or Compare Specific Conditions                  #
#                                                                             #
# #############################################################################

st.markdown("## 🔬 Part 2: Diagnosis Deep Dive")
st.markdown("*Explore a single diagnosis in detail, or compare two conditions side-by-side.*")

st.divider()

# =============================================================================
# MODE SELECTOR
# =============================================================================
analysis_mode = st.radio(
    "Choose analysis mode:",
    ["🔍 Explore One Diagnosis", "⚖️ Compare Two Diagnoses"],
    horizontal=True,
    help="Single mode for deep exploration; Compare mode for side-by-side analysis"
)

st.divider()

diagnosis_options = dx_counts['long_title'].tolist()

# =============================================================================
# SINGLE DIAGNOSIS MODE
# =============================================================================
if analysis_mode == "🔍 Explore One Diagnosis":
    
    st.header("🎯 Select a Diagnosis")
    
    default_index = 0
    if st.session_state.selected_diagnosis and st.session_state.selected_diagnosis in diagnosis_options:
        default_index = diagnosis_options.index(st.session_state.selected_diagnosis)
    
    selected_diagnosis = st.selectbox(
        "Choose a diagnosis to explore:",
        options=diagnosis_options,
        index=default_index,
        help="Choose a diagnosis to see detailed patient information. You can also click a bar in Part 1!"
    )
    
    st.session_state.selected_diagnosis = selected_diagnosis
    
    # Get stats
    stats = get_diagnosis_stats(selected_diagnosis, dx_counts, dx_with_names, patients, 
                                admissions, icustays, diagnosis_burden)
    
    st.info(f"📌 **{selected_diagnosis}** — {stats['n_patients']} patients")
    
    st.divider()
    
    # Demographics
    st.header("👥 Demographics")
    st.markdown("*Who are the patients with this diagnosis?*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_chart = alt.Chart(stats['patient_data']).mark_bar().encode(
            x=alt.X('anchor_age:Q', bin=alt.Bin(step=10), title='Age'),
            y=alt.Y('count()', title='Number of Patients'),
            tooltip=[alt.Tooltip('anchor_age:Q', bin=alt.Bin(step=10), title='Age Range'),
                     alt.Tooltip('count()', title='Patients')]
        ).properties(title='Age Distribution', height=250)
        st.altair_chart(age_chart, use_container_width=True)
    
    with col2:
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
    st.header("📋 Outcomes")
    st.markdown("*How do patients with this diagnosis fare?*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_mort = stats['mortality'] - overall_mortality
        st.metric("Mortality Rate", f"{stats['mortality']:.1f}%", 
                  delta=f"{delta_mort:+.1f}% vs overall", delta_color="inverse")
    with col2:
        st.metric("Average Age", f"{stats['avg_age']:.1f} years")
    with col3:
        st.metric("Admissions with ICU", f"{stats['icu_rate']:.0f}%")
    
    st.divider()
    
    # Comorbidities
    st.header("🔀 Top Comorbidities")
    st.markdown("*What other diagnoses do these patients have?*")
    
    comorbidities = get_comorbidities(dx_with_names, stats['patient_ids'], stats['icd_code'], top_n=10)
    
    if len(comorbidities) > 0:
        comorbidities['short_title'] = comorbidities['long_title'].apply(lambda x: x[:40] + '...' if len(str(x)) > 40 else x)
        
        comorbidity_chart = alt.Chart(comorbidities).mark_bar().encode(
            x=alt.X('patient_count:Q', title='Number of Patients'),
            y=alt.Y('short_title:N', title='Diagnosis', sort='-x'),
            tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('patient_count:Q', title='Patients')]
        ).properties(height=300)
        st.altair_chart(comorbidity_chart, use_container_width=True)
    else:
        st.info("No comorbidities found.")
    
    st.divider()
    
    # Labs & Medications
    st.header("🧪 Labs & Medications")
    st.markdown("*What lab patterns and treatments are common for these patients?*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Lab Value Abnormalities**")
        st.caption("Percentage of values outside reference range")
        
        lab_data, ref_ranges, top_labs = get_lab_distributions(labevents, d_labitems, stats['patient_ids'], top_n=6)
        
        if len(lab_data) > 0:
            lab_summary = ref_ranges.copy()
            lab_summary['pct_abnormal_display'] = (lab_summary['pct_abnormal'] * 100).round(0)
            lab_summary['short_label'] = lab_summary['label'].apply(lambda x: x[:12] + '...' if len(x) > 12 else x)
            lab_summary['has_ref_range'] = lab_summary['pct_abnormal'] > 0
            
            abnormal_chart = alt.Chart(lab_summary).mark_bar().encode(
                x=alt.X('pct_abnormal_display:Q', title='% Abnormal', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('short_label:N', title=None, 
                       sort=alt.EncodingSortField('n_measurements', order='descending'),
                       axis=alt.Axis(labelLimit=100)),
                color=alt.condition(alt.datum.has_ref_range,
                    alt.Color('pct_abnormal_display:Q', scale=alt.Scale(scheme='reds', domain=[0, 100]), legend=None),
                    alt.value('#666666')),
                tooltip=[alt.Tooltip('label:N', title='Lab'),
                         alt.Tooltip('n_measurements:Q', title='Measurements'),
                         alt.Tooltip('pct_abnormal_display:Q', title='% Abnormal')]
            ).properties(height=200)
            
            st.altair_chart(abnormal_chart, use_container_width=True)
            st.caption("Red = % abnormal values | Gray = no reference range available")
        else:
            st.info("No lab values available for these patients.")
    
    with col2:
        st.markdown("**Top Medications**")
        st.caption("Most frequently prescribed drugs")
        
        top_meds = get_top_meds_for_patients(prescriptions, stats['patient_ids'])
        
        if len(top_meds) > 0:
            top_meds = top_meds.head(5)
            top_meds['short_drug'] = top_meds['drug'].apply(lambda x: x[:15] + '...' if len(str(x)) > 15 else x)
            
            med_chart = alt.Chart(top_meds).mark_bar(color='#4C78A8').encode(
                x=alt.X('count:Q', title='Prescriptions'),
                y=alt.Y('short_drug:N', title=None, sort='-x', axis=alt.Axis(labelLimit=120, labelFontSize=11)),
                tooltip=[alt.Tooltip('drug:N', title='Medication'), alt.Tooltip('count:Q', title='Prescriptions')]
            ).properties(height=180)
            st.altair_chart(med_chart, use_container_width=True)
        else:
            st.info("No medication data available.")


# =============================================================================
# COMPARISON MODE
# =============================================================================
else:
    st.header("⚖️ Compare Two Diagnoses")
    st.markdown("*Select two conditions to see how their patient populations differ.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_a = 0
        if st.session_state.selected_diagnosis and st.session_state.selected_diagnosis in diagnosis_options:
            default_a = diagnosis_options.index(st.session_state.selected_diagnosis)
        
        diagnosis_a = st.selectbox(
            "🔵 Diagnosis A:",
            options=diagnosis_options,
            index=default_a,
            key="dx_a"
        )
    
    with col2:
        # Default to second most common diagnosis to avoid same selection
        default_b = 1 if default_a == 0 else 0
        diagnosis_b = st.selectbox(
            "🟠 Diagnosis B:",
            options=diagnosis_options,
            index=default_b,
            key="dx_b"
        )
    
    if diagnosis_a == diagnosis_b:
        st.warning("⚠️ Please select two different diagnoses to compare.")
        st.stop()
    
    # Get stats for both
    stats_a = get_diagnosis_stats(diagnosis_a, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden)
    stats_b = get_diagnosis_stats(diagnosis_b, dx_counts, dx_with_names, patients, admissions, icustays, diagnosis_burden)
    
    # Color scheme
    color_a = '#4C78A8'  # Blue
    color_b = '#F58518'  # Orange
    
    st.divider()
    
    # ==========================================================================
    # COMPARISON: Key Metrics
    # ==========================================================================
    st.subheader("📊 Key Metrics Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🔵 {smart_truncate(diagnosis_a, 45)}**")
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.metric("Patients", stats_a['n_patients'])
            st.metric("Avg Age", f"{stats_a['avg_age']:.1f} yrs")
        with subcol2:
            st.metric("Mortality", f"{stats_a['mortality']:.1f}%")
            st.metric("ICU Rate", f"{stats_a['icu_rate']:.0f}%")
        with subcol3:
            st.metric("Median Dx", f"{stats_a['median_burden']:.0f}")
            st.metric("% Female", f"{stats_a['pct_female']:.0f}%")
    
    with col2:
        st.markdown(f"**🟠 {smart_truncate(diagnosis_b, 45)}**")
        # Calculate deltas (B relative to A)
        delta_patients = stats_b['n_patients'] - stats_a['n_patients']
        delta_mort = stats_b['mortality'] - stats_a['mortality']
        delta_age = stats_b['avg_age'] - stats_a['avg_age']
        delta_icu = stats_b['icu_rate'] - stats_a['icu_rate']
        delta_burden = stats_b['median_burden'] - stats_a['median_burden']
        delta_female = stats_b['pct_female'] - stats_a['pct_female']
        
        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.metric("Patients", stats_b['n_patients'], delta=f"{delta_patients:+d}")
            st.metric("Avg Age", f"{stats_b['avg_age']:.1f} yrs", delta=f"{delta_age:+.1f}")
        with subcol2:
            st.metric("Mortality", f"{stats_b['mortality']:.1f}%", delta=f"{delta_mort:+.1f}%", delta_color="inverse")
            st.metric("ICU Rate", f"{stats_b['icu_rate']:.0f}%", delta=f"{delta_icu:+.0f}%", delta_color="inverse")
        with subcol3:
            st.metric("Median Dx", f"{stats_b['median_burden']:.0f}", delta=f"{delta_burden:+.0f}")
            st.metric("% Female", f"{stats_b['pct_female']:.0f}%", delta=f"{delta_female:+.0f}%")
    
    st.divider()
    
    # ==========================================================================
    # COMPARISON: Age Distribution
    # ==========================================================================
    st.subheader("👥 Age Distribution")
    
    # Combine patient data for overlay chart
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
    
    # ==========================================================================
    # COMPARISON: Gender Distribution
    # ==========================================================================
    st.subheader("👫 Gender Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_a = stats_a['patient_data']['gender'].value_counts().reset_index()
        gender_a.columns = ['gender', 'count']
        gender_a['pct'] = (gender_a['count'] / gender_a['count'].sum() * 100).round(1)
        
        gender_chart_a = alt.Chart(gender_a).mark_bar(color=color_a).encode(
            x=alt.X('gender:N', title='Gender'),
            y=alt.Y('pct:Q', title='Percentage', scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip('gender:N'), alt.Tooltip('count:Q', title='Patients'), alt.Tooltip('pct:Q', format='.1f', title='%')]
        ).properties(title=f'🔵 {smart_truncate(diagnosis_a, 30)}', height=200)
        st.altair_chart(gender_chart_a, use_container_width=True)
    
    with col2:
        gender_b = stats_b['patient_data']['gender'].value_counts().reset_index()
        gender_b.columns = ['gender', 'count']
        gender_b['pct'] = (gender_b['count'] / gender_b['count'].sum() * 100).round(1)
        
        gender_chart_b = alt.Chart(gender_b).mark_bar(color=color_b).encode(
            x=alt.X('gender:N', title='Gender'),
            y=alt.Y('pct:Q', title='Percentage', scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip('gender:N'), alt.Tooltip('count:Q', title='Patients'), alt.Tooltip('pct:Q', format='.1f', title='%')]
        ).properties(title=f'🟠 {smart_truncate(diagnosis_b, 30)}', height=200)
        st.altair_chart(gender_chart_b, use_container_width=True)
    
    st.divider()
    
    # ==========================================================================
    # COMPARISON: Top Comorbidities
    # ==========================================================================
    st.subheader("🔀 Top Comorbidities")
    st.markdown("*What other conditions do these patient groups have?*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption(f"**🔵 {smart_truncate(diagnosis_a, 35)}**")
        comorb_a = get_comorbidities_for_comparison(dx_with_names, stats_a['patient_ids'], 
                                                     [stats_a['icd_code'], stats_b['icd_code']], top_n=8)
        if len(comorb_a) > 0:
            comorb_a['short_title'] = comorb_a['long_title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            chart_a = alt.Chart(comorb_a).mark_bar(color=color_a).encode(
                x=alt.X('patient_count:Q', title='Patients'),
                y=alt.Y('short_title:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('patient_count:Q', title='Patients')]
            ).properties(height=250)
            st.altair_chart(chart_a, use_container_width=True)
    
    with col2:
        st.caption(f"**🟠 {smart_truncate(diagnosis_b, 35)}**")
        comorb_b = get_comorbidities_for_comparison(dx_with_names, stats_b['patient_ids'], 
                                                     [stats_a['icd_code'], stats_b['icd_code']], top_n=8)
        if len(comorb_b) > 0:
            comorb_b['short_title'] = comorb_b['long_title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
            chart_b = alt.Chart(comorb_b).mark_bar(color=color_b).encode(
                x=alt.X('patient_count:Q', title='Patients'),
                y=alt.Y('short_title:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('long_title:N', title='Diagnosis'), alt.Tooltip('patient_count:Q', title='Patients')]
            ).properties(height=250)
            st.altair_chart(chart_b, use_container_width=True)
    
    # Shared comorbidities callout
    if len(comorb_a) > 0 and len(comorb_b) > 0:
        shared = set(comorb_a['icd_code']) & set(comorb_b['icd_code'])
        if shared:
            shared_names = comorb_a[comorb_a['icd_code'].isin(shared)]['long_title'].tolist()[:3]
            st.info(f"🔗 **Shared comorbidities:** {', '.join([smart_truncate(n, 30) for n in shared_names])}")
    
    st.divider()
    
    # ==========================================================================
    # COMPARISON: Top Medications
    # ==========================================================================
    st.subheader("💊 Top Medications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption(f"**🔵 {smart_truncate(diagnosis_a, 35)}**")
        meds_a = get_top_meds_for_patients(prescriptions, stats_a['patient_ids']).head(6)
        if len(meds_a) > 0:
            meds_a['short_drug'] = meds_a['drug'].apply(lambda x: x[:18] + '...' if len(x) > 18 else x)
            med_chart_a = alt.Chart(meds_a).mark_bar(color=color_a).encode(
                x=alt.X('count:Q', title='Prescriptions'),
                y=alt.Y('short_drug:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('drug:N', title='Medication'), alt.Tooltip('count:Q', title='Prescriptions')]
            ).properties(height=200)
            st.altair_chart(med_chart_a, use_container_width=True)
        else:
            st.info("No medication data.")
    
    with col2:
        st.caption(f"**🟠 {smart_truncate(diagnosis_b, 35)}**")
        meds_b = get_top_meds_for_patients(prescriptions, stats_b['patient_ids']).head(6)
        if len(meds_b) > 0:
            meds_b['short_drug'] = meds_b['drug'].apply(lambda x: x[:18] + '...' if len(x) > 18 else x)
            med_chart_b = alt.Chart(meds_b).mark_bar(color=color_b).encode(
                x=alt.X('count:Q', title='Prescriptions'),
                y=alt.Y('short_drug:N', title=None, sort='-x'),
                tooltip=[alt.Tooltip('drug:N', title='Medication'), alt.Tooltip('count:Q', title='Prescriptions')]
            ).properties(height=200)
            st.altair_chart(med_chart_b, use_container_width=True)
        else:
            st.info("No medication data.")
    
    st.divider()
    
    # ==========================================================================
    # COMPARISON: Patient Overlap
    # ==========================================================================
    st.subheader("🔄 Patient Overlap")
    
    set_a = set(stats_a['patient_ids'])
    set_b = set(stats_b['patient_ids'])
    overlap = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Only {smart_truncate(diagnosis_a, 20)}", len(only_a))
    with col2:
        st.metric("Have Both", len(overlap))
    with col3:
        st.metric(f"Only {smart_truncate(diagnosis_b, 20)}", len(only_b))
    
    # Simple overlap visualization
    overlap_data = pd.DataFrame({
        'category': [f'Only 🔵', 'Both', f'Only 🟠'],
        'count': [len(only_a), len(overlap), len(only_b)],
        'color': [color_a, '#72B7B2', color_b]
    })
    
    overlap_chart = alt.Chart(overlap_data).mark_bar().encode(
        x=alt.X('category:N', title=None, sort=[f'Only 🔵', 'Both', f'Only 🟠']),
        y=alt.Y('count:Q', title='Patients'),
        color=alt.Color('category:N', scale=alt.Scale(domain=[f'Only 🔵', 'Both', f'Only 🟠'], 
                                                       range=[color_a, '#72B7B2', color_b]), legend=None),
        tooltip=[alt.Tooltip('category:N', title='Group'), alt.Tooltip('count:Q', title='Patients')]
    ).properties(height=200)
    
    st.altair_chart(overlap_chart, use_container_width=True)
    
    if len(overlap) > 0:
        pct_overlap_a = len(overlap) / len(set_a) * 100
        pct_overlap_b = len(overlap) / len(set_b) * 100
        st.caption(f"💡 {len(overlap)} patients have both diagnoses ({pct_overlap_a:.0f}% of 🔵, {pct_overlap_b:.0f}% of 🟠)")


st.divider()


# =============================================================================
# DATA PREVIEW (Development Tool)
# =============================================================================
with st.expander("🔍 Preview Raw Data (click to expand)"):
    table_choice = st.selectbox(
        "Select a table to preview:",
        ["patients", "admissions", "diagnoses_icd", "d_icd_diagnoses", 
         "labevents", "d_labitems", "prescriptions", "icustays", "dx_counts (processed)"]
    )
    
    if table_choice == "patients":
        st.dataframe(patients.head(10))
    elif table_choice == "admissions":
        st.dataframe(admissions.head(10))
    elif table_choice == "diagnoses_icd":
        st.dataframe(diagnoses.head(10))
    elif table_choice == "d_icd_diagnoses":
        st.dataframe(d_icd.head(10))
    elif table_choice == "labevents":
        st.dataframe(labevents.head(10))
    elif table_choice == "d_labitems":
        st.dataframe(d_labitems.head(10))
    elif table_choice == "prescriptions":
        st.dataframe(prescriptions.head(10))
    elif table_choice == "icustays":
        st.dataframe(icustays.head(10))
    elif table_choice == "dx_counts (processed)":
        st.dataframe(dx_counts.head(20))
