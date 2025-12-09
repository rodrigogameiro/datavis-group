# pages/02_Outcomes.py
# Streamlit page — Outcomes & Mortality (MIMIC-IV demo, CSV-only)

import os, re, textwrap
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

PAGE_TITLE = "Outcomes & Mortality (CSV only)"
SHOW_ICU = True  # set False to hide the ICU section

# Avoid blank charts from row caps
alt.data_transformers.disable_max_rows()

# ---------------------------- Utils ----------------------------
def _wrap_label(s: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width)) if isinstance(s, str) else str(s)

# ---------------------- Data Loader (CSV only) ------------------
@st.cache_data(show_spinner=True)
def load_csv(csv_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (hosp_df, icu_df) from local CSV files."""
    assert csv_dir, "CSV directory not provided."

    admissions = pd.read_csv(
        os.path.join(csv_dir, "admissions.csv"),
        parse_dates=["admittime", "dischtime", "deathtime"],
        low_memory=False,
    )
    patients = pd.read_csv(
        os.path.join(csv_dir, "patients.csv"),
        parse_dates=["dod"],
        low_memory=False,
    )
    icustays = pd.read_csv(
        os.path.join(csv_dir, "icustays.csv"),
        parse_dates=["intime", "outtime"],
        low_memory=False,
    )

    # Hospital table with LOS
    hosp_df = (
        admissions.merge(patients, on="subject_id", how="inner")
        .assign(hosp_los_days=lambda d: (d["dischtime"] - d["admittime"]).dt.total_seconds() / 86400.0)
    )

    # ICU LOS
    icustays["icu_los_days"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 86400.0
    icu_df = icustays.copy()

    # ---- Race/Ethnicity grouping ----
    def _race_group_from_text(raw: str) -> str:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return "Other/Unknown"
        t = str(raw).upper().strip()
        t = re.sub(r"[\/,\-]+", " ", t)
        t = re.sub(r"\s+", " ", t)
        if any(k in t for k in ["UNKNOWN", "DECLINED", "UNABLE", "NOT SPECIFIED", "REFUSED"]):
            return "Other/Unknown"
        if any(k in t for k in ["HISPANIC", "LATINO", "LATINA", "LATINX"]):
            return "Hispanic/Latino"
        if ("BLACK" in t) or ("AFRICAN" in t and "AMERICAN" in t) or ("AFRICAN" in t and "BLACK" in t):
            return "Black"
        if "ASIAN" in t:
            return "Asian"
        if "WHITE" in t:
            return "White"
        if "PORTUGUESE" in t:
            return "Other/Unknown"
        return "Other/Unknown"

    if "race" in hosp_df.columns and hosp_df["race"].notna().any():
        hosp_df["race_group"] = hosp_df["race"].apply(_race_group_from_text)
    elif "ethnicity" in hosp_df.columns and hosp_df["ethnicity"].notna().any():
        hosp_df["race_group"] = hosp_df["ethnicity"].apply(_race_group_from_text)
    else:
        hosp_df["race_group"] = "Other/Unknown"

    # ICU mortality within stay window
    icu_full = icu_df.merge(
        hosp_df[["subject_id", "hadm_id", "deathtime", "dod"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    def died_in_icu(row) -> int:
        death_t = row.get("deathtime") if pd.notnull(row.get("deathtime")) else row.get("dod")
        if pd.isna(death_t) or pd.isna(row.get("intime")) or pd.isna(row.get("outtime")):
            return 0
        return int(row["intime"] <= death_t <= row["outtime"])

    icu_full["icu_expire_flag"] = icu_full.apply(died_in_icu, axis=1)

    # 30-day post-discharge mortality
    def died_30d(discharge, dod):
        if pd.isna(discharge) or pd.isna(dod):
            return 0
        return int(dod <= discharge + pd.Timedelta(days=30))

    hosp_df["mortality_30d"] = [died_30d(d, o) for d, o in zip(hosp_df["dischtime"], hosp_df["dod"])]

    # normalize insurance text
    if "insurance" in hosp_df.columns:
        hosp_df["insurance"] = hosp_df["insurance"].fillna("Unknown").astype(str)
    else:
        hosp_df["insurance"] = "Unknown"

    return hosp_df, icu_full

# ----------------------------- Page -----------------------------
def render():
    st.header(PAGE_TITLE)
    st.caption("MIMIC-IV demo — hospital & ICU outcomes, mortality, LOS (CSV backend)")

    # Sidebar (CSV only) — auto-detect directory
    with st.sidebar:
        st.subheader("Data Source (CSV only)")

        def _guess_csv_dir() -> str:
            # priority: secrets -> env var -> ./data if files exist -> ""
            try:
                from_secret = (st.secrets.get("csv", {}) or {}).get("dir")
            except Exception:
                from_secret = None
            from_env = os.getenv("MIMIC_CSV_DIR")
            if from_secret:
                return from_secret
            if from_env:
                return from_env
            maybe = "data"
            need = ["admissions.csv", "patients.csv", "icustays.csv"]
            if all(os.path.exists(os.path.join(maybe, f)) for f in need):
                return maybe
            return ""

        csv_dir = st.text_input(
            "CSV directory (admissions.csv / patients.csv / icustays.csv)",
            value=_guess_csv_dir()
        )

    # Optional GitHub link
    repo_url = os.getenv("GITHUB_REPO_URL", "")
    try:
        if not repo_url:
            repo_url = (st.secrets.get("github", {}) or {}).get("repo_url", "")
    except Exception:
        pass
    if repo_url:
        st.link_button("View Source on GitHub", repo_url)

    # Load data
    try:
        hosp_df, icu_df = load_csv(csv_dir)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # ---------------- KPIs ----------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Admissions", int(hosp_df["hadm_id"].nunique()))
    with c2:
        st.metric("In-hospital Mortality", f"{hosp_df['hospital_expire_flag'].fillna(0).astype(int).mean():.1%}")
    with c3:
        st.metric("Median Hospital LOS (d)", f"{hosp_df['hosp_los_days'].median():.2f}")
    with c4:
        icu_mort = icu_df["icu_expire_flag"].mean() if len(icu_df) else 0.0
        st.metric("ICU Mortality (stay-window)", f"{icu_mort:.1%}")

    # ---------------- Hospital Outcomes ----------------
    st.subheader("Hospital Outcomes")

    # Discharge destination — Pie
    st.markdown("### Discharge Destination")
    all_locs = sorted(hosp_df["discharge_location"].fillna("Unknown").astype(str).unique())
    sel_locs = st.multiselect("Select destinations to include", all_locs, default=all_locs, key="locs_include")

    disch_src = hosp_df[hosp_df["discharge_location"].fillna("Unknown").astype(str).isin(sel_locs)]
    disch = (
        disch_src["discharge_location"].fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("discharge_location")
        .reset_index(name="count")
    )
    disch["pct"] = disch["count"] / max(1, disch["count"].sum())

    pie = (
        alt.Chart(disch)
        .mark_arc(outerRadius=140)
        .encode(
            theta=alt.Theta("pct:Q", stack=True, title=None),
            color=alt.Color("discharge_location:N", title="Destination"),
            tooltip=[
                alt.Tooltip("discharge_location:N", title="Destination"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(pie, use_container_width=True)

    # Hospital LOS distribution — single chart (no clipping)
    st.markdown("### Hospital LOS distribution (days)")
    los_df = (
        hosp_df.assign(los_day=lambda d: d["hosp_los_days"].round().astype(int))
               .groupby("los_day")["hadm_id"].count().reset_index(name="cases")
    )
    los_df["cases"] = pd.to_numeric(los_df["cases"], errors="coerce").fillna(0).astype(float)

    los_chart = (
        alt.Chart(los_df)
        .mark_bar()
        .encode(
            x=alt.X("los_day:O", title="LOS (days)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("cases:Q", title="Count"),
            tooltip=[alt.Tooltip("los_day:O", title="LOS (d)"),
                     alt.Tooltip("cases:Q", title="Count")],
        )
        .properties(height=320)
    )
    st.altair_chart(los_chart, use_container_width=True)

    los = hosp_df["hosp_los_days"].dropna()
    st.caption(
        f"median={los.median():.1f} d, IQR={los.quantile(0.25):.1f}–{los.quantile(0.75):.1f} d, "
        f"95th={los.quantile(0.95):.1f} d"
    )

    # In-hospital Mortality by Insurance (rate only)
    st.markdown("### In-hospital Mortality by Insurance")
    ins_all = sorted(hosp_df["insurance"].astype(str).unique())
    ins_sel = st.multiselect("Insurance types to include", ins_all, default=ins_all, key="ins_include")
    tmp_ins = hosp_df[hosp_df["insurance"].astype(str).isin(ins_sel)].assign(
        hosp_mort=lambda d: d["hospital_expire_flag"].fillna(0).astype(int)
    )
    by_ins = tmp_ins.groupby("insurance", dropna=False)["hosp_mort"].mean().reset_index()
    by_ins["label"] = by_ins["insurance"].astype(str).apply(lambda s: _wrap_label(s, 12))
    ins_m_chart = (
        alt.Chart(by_ins.sort_values("hosp_mort", ascending=True))
        .mark_bar()
        .encode(
            y=alt.Y("label:N", title="Insurance", sort=None,
                    axis=alt.Axis(labelLimit=1000, labelPadding=10)),
            x=alt.X("hosp_mort:Q", title="Mortality", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("insurance:N", title="Insurance"),
                     alt.Tooltip("hosp_mort:Q", format=".1%", title="Mortality")],
        ).properties(height=max(240, 26 * len(by_ins)), padding={"left": 14})
    )
    st.altair_chart(ins_m_chart, use_container_width=True)

    # ---------------- Mortality by Race / Ethnicity — rate only ----------------
    st.markdown("### In-hospital Mortality by Race / Ethnicity")
    race_all = sorted(hosp_df["race_group"].astype(str).unique())
    race_sel = st.multiselect("Race / Ethnicity to include", race_all, default=race_all, key="race_include")

    tmp = hosp_df[hosp_df["race_group"].astype(str).isin(race_sel)].assign(
        hosp_mort=lambda d: d["hospital_expire_flag"].fillna(0).astype(int)
    )
    agg = (
        tmp.groupby("race_group", dropna=False)
           .agg(deaths=("hosp_mort", "sum"), n=("hosp_mort", "count"))
           .reset_index()
    )
    agg["mortality"] = agg["deaths"] / agg["n"].replace(0, np.nan)
    agg["label"]     = agg["race_group"].astype(str).apply(lambda s: _wrap_label(s, 12))
    agg = agg.sort_values(["mortality", "n"], ascending=[False, False]).reset_index(drop=True)

    overall_rate = (agg["deaths"].sum() / max(1, agg["n"].sum())) if len(agg) else 0.0
    agg["text"] = (agg["mortality"].fillna(0).map(lambda v: f"{v*100:.1f}%")
                   + "  (" + agg["deaths"].astype(str) + "/" + agg["n"].astype(str) + ")")

    base = alt.Chart(agg).properties(height=max(240, 26 * len(agg)))
    rule = alt.Chart(pd.DataFrame({"overall":[overall_rate]})).mark_rule(strokeDash=[6,4]).encode(
        x=alt.X("overall:Q", axis=alt.Axis(format="%"), title="Mortality")
    )
    bars = base.mark_bar().encode(
        y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=1000, labelPadding=10)),
        x=alt.X("mortality:Q", title="Mortality", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip("race_group:N", title="Race"),
            alt.Tooltip("deaths:Q",     title="Deaths"),
            alt.Tooltip("n:Q",          title="N"),
            alt.Tooltip("mortality:Q",  title="Rate", format=".2%")
        ],
    )
    labels = base.mark_text(align="left", dx=5).encode(
        y=alt.Y("label:N", sort=None),
        x="mortality:Q",
        text=alt.Text("text:N"),
    )
    st.altair_chart((bars + labels + rule).properties(padding={"left": 14}), use_container_width=True)
    st.caption("Dashed line = overall in-hospital mortality.")

    st.divider()

    # -------------------------- ICU Outcomes --------------------------
    if SHOW_ICU:
        st.subheader("ICU Outcomes")

        if len(icu_df):
            icu_df = icu_df.copy()
            icu_df["first_careunit"] = icu_df.get("first_careunit").fillna("Unknown").astype(str)

        all_units = sorted(icu_df["first_careunit"].unique()) if len(icu_df) else []
        unit_sel  = st.multiselect("ICU units to include", all_units, default=all_units, key="icu_units")
        icu_use   = icu_df[icu_df["first_careunit"].isin(unit_sel)] if (len(icu_df) and unit_sel) else icu_df

        # ICU LOS (median)
        st.markdown("### ICU LOS (median) by unit")
        if len(icu_use):
            los_by_unit = icu_use.groupby("first_careunit", dropna=False)["icu_los_days"].median().reset_index()
            los_by_unit["label"] = los_by_unit["first_careunit"].apply(lambda x: _wrap_label(x, 22))
            df = los_by_unit.sort_values("icu_los_days", ascending=True)
            chart1 = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    y=alt.Y("label:N", title=None, sort=None,
                            axis=alt.Axis(labelLimit=1000, labelPadding=12)),
                    x=alt.X("icu_los_days:Q", title="Median days"),
                    tooltip=[alt.Tooltip("first_careunit:N", title="Unit"),
                             alt.Tooltip("icu_los_days:Q", title="Median LOS (days)")],
                )
                .properties(height=max(260, 26 * len(df)), padding={"left": 16})
            )
            st.altair_chart(chart1, use_container_width=True)
        else:
            st.info("No ICU stays in selected units.")

        st.divider()

        # ICU Mortality (rate only)
        st.markdown("### ICU Mortality by unit")
        if len(icu_use):
            g2 = (
                icu_use.groupby("first_careunit", dropna=False)
                       .agg(deaths=("icu_expire_flag","sum"),
                            n=("icu_expire_flag","count"))
                       .reset_index()
            )
            g2["mortality"] = g2["deaths"] / g2["n"].replace(0, np.nan)
            g2["label"]     = g2["first_careunit"].apply(lambda x: _wrap_label(x, 22))
            g2 = g2.sort_values(["mortality","n"], ascending=[False, False]).reset_index(drop=True)

            overall_icu = (g2["deaths"].sum() / max(1, g2["n"].sum())) if len(g2) else 0.0
            g2["text"] = (g2["mortality"].fillna(0).map(lambda v: f"{v*100:.1f}%")
                          + "  (" + g2["deaths"].astype(str) + "/" + g2["n"].astype(str) + ")")

            base_i = alt.Chart(g2).properties(height=max(260, 26 * len(g2)))
            rule_i = alt.Chart(pd.DataFrame({"overall":[overall_icu]})).mark_rule(strokeDash=[6,4]).encode(
                x=alt.X("overall:Q", axis=alt.Axis(format="%"), title="Proportion")
            )
            bars_i = base_i.mark_bar().encode(
                y=alt.Y("label:N", title=None, sort=None,
                        axis=alt.Axis(labelLimit=1000, labelPadding=12)),
                x=alt.X("mortality:Q", title="Proportion", axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("first_careunit:N", title="Unit"),
                    alt.Tooltip("deaths:Q",        title="Deaths"),
                    alt.Tooltip("n:Q",             title="N"),
                    alt.Tooltip("mortality:Q",     title="Rate", format=".2%"),
                ],
            )
            labels_i = base_i.mark_text(align="left", dx=5).encode(
                y=alt.Y("label:N", sort=None),
                x="mortality:Q",
                text=alt.Text("text:N"),
            )

            st.altair_chart((bars_i + labels_i + rule_i).properties(padding={"left": 16}), use_container_width=True)
            st.caption(f"Overall ICU mortality (stay-window) across selected units: {overall_icu:.1%}.")
        else:
            st.info("No ICU stays in selected units.")

# --------- Run ----------
render()








