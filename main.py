import traceback

import pandas as pd
import streamlit as st

from backend.data import load_csv, prepare_ranked_df, validate_columns
from backend.detect import run_detection
from backend.explain import run_explanations
from from_list_to_shapy_values import string2num


def init_session_state():
    if "detection_state" not in st.session_state:
        st.session_state.detection_state = None
    if "explanation_df" not in st.session_state:
        st.session_state.explanation_df = None


def decode_pattern(pattern_str, attributes):
    decoded = string2num(pattern_str)
    row = {}
    for idx, attr in enumerate(attributes):
        value = decoded[idx] if idx < len(decoded) else -1
        row[attr] = "ANY" if value == -1 else value
    return row


def build_value_label_maps(df, attributes):
    label_maps = {}

    for attr in attributes:
        source_col = None
        if attr.endswith("_C"):
            candidate = attr[:-2]
            if candidate in df.columns:
                source_col = candidate
        elif attr.endswith("_binary"):
            candidate = attr[:-7]
            if candidate in df.columns:
                source_col = candidate

        if source_col is None:
            continue

        pairs = df[[attr, source_col]].dropna().drop_duplicates()
        value_map = {}
        for _, rec in pairs.iterrows():
            try:
                encoded_val = int(rec[attr])
            except Exception:
                continue
            value_map[encoded_val] = str(rec[source_col])

        if value_map:
            label_maps[attr] = value_map

    return label_maps


def apply_label_maps(decoded_df, attributes, label_maps):
    pretty_df = decoded_df.copy()

    for attr in attributes:
        if attr not in pretty_df.columns or attr not in label_maps:
            continue

        mapping = label_maps[attr]

        def to_label(value):
            if value == "ANY":
                return "ANY"
            try:
                code = int(value)
            except Exception:
                return str(value)

            label = mapping.get(code)
            if label is None:
                return str(value)
            return f"{label} ({code})"

        pretty_df[attr] = pretty_df[attr].apply(to_label)

    return pretty_df


def to_arrow_safe_df(df):
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_object_dtype(display_df[col]):
            display_df[col] = display_df[col].astype(str)
    return display_df


def pattern_to_text(pattern_str, attributes, label_maps):
    decoded_df = pd.DataFrame([decode_pattern(pattern_str, attributes)])
    pretty_df = apply_label_maps(decoded_df, attributes, label_maps)
    row = pretty_df.iloc[0]
    return ", ".join([f"{attr}={row[attr]}" for attr in attributes])


def flatten_detected_groups(groups, attributes, k_min):
    rows = []
    for i, level in enumerate(groups):
        k_value = int(k_min) + i
        for pattern in level:
            rows.append({"k": k_value, **decode_pattern(pattern, attributes), "pattern": pattern})
    return rows


def build_explanation_table(explanation, group_patterns, selected_k, attributes, label_maps):
    explanation_values = list(explanation.values())
    rows = []

    for idx, pattern_str in enumerate(group_patterns):
        expected_key = int(selected_k) + idx
        group_records = explanation.get(expected_key)

        if group_records is None and idx < len(explanation_values):
            group_records = explanation_values[idx]
        if group_records is None:
            continue

        readable_group = pattern_to_text(pattern_str, attributes, label_maps)
        for record in group_records:
            feature_name = record.get("Attribute")
            shap_value = record.get("Shapley values")
            try:
                shap_num = float(shap_value)
            except Exception:
                continue

            rows.append(
                {
                    "selected_k": int(selected_k),
                    "group_pattern": pattern_str,
                    "group": readable_group,
                    "feature": feature_name,
                    "shapley_value": shap_num,
                }
            )

    if not rows:
        return pd.DataFrame()

    tidy_df = pd.DataFrame(rows)
    tidy_df["abs_shapley"] = tidy_df["shapley_value"].abs()
    tidy_df = tidy_df.sort_values(by=["group_pattern", "abs_shapley"], ascending=[True, False]).reset_index(drop=True)
    tidy_df["impact_rank"] = tidy_df.groupby("group_pattern").cumcount() + 1
    tidy_df["shapley_value"] = tidy_df["shapley_value"].round(4)
    tidy_df["abs_shapley"] = tidy_df["abs_shapley"].round(4)

    return tidy_df[
        [
            "selected_k",
            "group_pattern",
            "group",
            "feature",
            "shapley_value",
            "abs_shapley",
            "impact_rank",
        ]
    ]


st.set_page_config(page_title="Biased Representation Demo", layout="wide")
st.title("Detection of Groups with Biased Representation in Ranking")
init_session_state()

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

left_col, right_col = st.columns(2)

with left_col:
    rank_col = st.text_input("Rank column", value="rank")
    threshold = st.number_input("Size threshold (Thc)", value=50.0)
    mode = st.selectbox("Mode", ["prop", "global"])

with right_col:
    k_min = st.number_input("k_min", value=10, min_value=1)
    k_max = st.number_input("k_max", value=15, min_value=1)
    if mode == "prop":
        alpha = st.number_input("Alpha", value=0.05, format="%.4f")
    else:
        alpha = st.number_input("Global lower bound", value=10.0, step=1.0)

df = None
rank_valid = False
selected_attributes = []
possible_attrs = []

if uploaded is not None:
    df = load_csv(uploaded_file=uploaded)
    st.write("Preview")
    st.dataframe(to_arrow_safe_df(df.head()))

    if rank_col not in df.columns:
        st.error(f"Rank column '{rank_col}' was not found in the CSV.")
    else:
        rank_valid = True
        possible_attrs = [c for c in df.columns if c != rank_col]
        recommended_attrs = [
            c
            for c in possible_attrs
            if c.endswith("_C") or c.endswith("_binary") or pd.api.types.is_numeric_dtype(df[c])
        ]
        default_attrs = recommended_attrs[:3] if recommended_attrs else possible_attrs[:3]
        selected_attributes = st.multiselect("Protected attributes", possible_attrs, default=default_attrs)

ready_to_run = uploaded is not None and rank_valid and len(selected_attributes) > 0

if st.button("Run detection", disabled=not ready_to_run):
    if not ready_to_run:
        st.error("Upload data, provide a valid rank column, and select at least one protected attribute.")
    else:
        try:
            if k_min > k_max:
                st.error("k_min must be less than or equal to k_max.")
                st.stop()

            ranked_df = prepare_ranked_df(df, rank_col)
            validate_columns(ranked_df, selected_attributes)
            label_maps = build_value_label_maps(ranked_df, selected_attributes)

            non_numeric = [c for c in selected_attributes if not pd.api.types.is_numeric_dtype(ranked_df[c])]
            if non_numeric:
                st.error(
                    "Selected attributes must be numeric/integer encoded for this implementation. "
                    f"Please choose encoded columns like *_C or *_binary. Invalid: {non_numeric}"
                )
                st.stop()

            detection_df = ranked_df[selected_attributes].copy()
            detection = run_detection(
                detection_df,
                selected_attributes,
                threshold,
                alpha,
                int(k_min),
                int(k_max),
                mode=mode,
            )

            st.success("Detection finished")
            st.session_state.detection_state = {
                "groups": detection["groups"],
                "selected_attributes": selected_attributes.copy(),
                "k_min": int(k_min),
                "k_max": int(k_max),
                "ranked_df": ranked_df,
                "value_label_maps": label_maps,
            }
            st.session_state.explanation_df = None

        except Exception:
            st.code(traceback.format_exc())

detection_state = st.session_state.detection_state
if detection_state is not None:
    groups = detection_state["groups"]
    attributes = detection_state["selected_attributes"]
    k_min_state = detection_state["k_min"]
    k_max_state = detection_state["k_max"]
    ranked_df = detection_state["ranked_df"]
    label_maps = detection_state["value_label_maps"]

    flat_rows = flatten_detected_groups(groups, attributes, k_min_state)

    if not flat_rows:
        st.info("No biased groups found.")
    else:
        st.subheader("Detected groups")
        decoded_df = pd.DataFrame(flat_rows)
        decoded_pretty_df = apply_label_maps(decoded_df, attributes, label_maps)
        decoded_pretty_df["pattern"] = decoded_pretty_df["pattern"].apply(
            lambda pattern_str: pattern_to_text(pattern_str, attributes, label_maps)
        )
        display_cols = ["k"] + attributes + ["pattern"]
        st.dataframe(to_arrow_safe_df(decoded_pretty_df[display_cols]))

        available_ks = list(range(int(k_min_state), int(k_max_state)))
        selected_k = st.selectbox("Select k for explanation", available_ks, key="selected_k_expl")
        level_index = int(selected_k) - int(k_min_state)

        if st.button("Run explanation for selected k", key="run_expl_btn"):
            with st.spinner("Running explanation... this can take a few seconds"):
                selected_groups = groups[level_index]
                if isinstance(selected_groups, set):
                    selected_group_patterns = sorted(list(selected_groups))
                else:
                    selected_group_patterns = list(selected_groups)

                explanation = run_explanations(ranked_df, selected_group_patterns, attributes, int(selected_k))
                st.session_state.explanation_df = build_explanation_table(
                    explanation,
                    selected_group_patterns,
                    selected_k,
                    attributes,
                    label_maps,
                )

            st.success(f"Explanation finished for k={selected_k}.")

        explanation_df = st.session_state.explanation_df
        if explanation_df is not None:
            if explanation_df.empty:
                st.info("No explanation rows returned for this k.")
            else:
                st.subheader("Explanation (Shapley values)")
                st.caption("Higher absolute Shapley values indicate stronger impact for that detected group.")

                patterns = explanation_df["group_pattern"].dropna().unique().tolist()
                for pattern in patterns:
                    pattern_df = explanation_df[explanation_df["group_pattern"] == pattern].copy()
                    pattern_df = pattern_df.sort_values(by="impact_rank")
                    readable_pattern = pattern_df["group"].iloc[0] if not pattern_df.empty else pattern

                    st.markdown(f"#### Pattern: {readable_pattern}")

                    table_df = pattern_df.drop(columns=["group_pattern", "group"], errors="ignore")
                    st.dataframe(to_arrow_safe_df(table_df))
