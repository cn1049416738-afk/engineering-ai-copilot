import os
import sys
import re
import hashlib

import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PATH SETUP
# ============================================================

sys.path.append("src")


# ============================================================
# LOCAL MODULE IMPORTS
# ============================================================

from pdf_reader import load_pdf_chunks
from embeddings import load_or_create_embeddings
from retriever import retrieve_top_chunks

from data_analyzer import (
    load_csv,
    get_basic_summary,
    get_numeric_columns,
    get_correlation_matrix,
    get_target_correlations,
    run_linear_regression,
)

from torch_model import (
    train_surrogate_model,
    predict_surrogate,
    check_out_of_distribution,
)


# ============================================================
# ENVIRONMENT / OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "OPENAI_API_KEY was not found. "
        "Please check your .env file or Streamlit Secrets."
    )
    st.stop()

client = OpenAI(
    api_key=api_key
)

os.makedirs(
    "data",
    exist_ok=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_answer_with_latex(text):

    parts = re.split(
        r"(\\\[[\s\S]*?\\\])",
        text,
    )

    for part in parts:

        if not part.strip():
            continue

        if (
            part.startswith(r"\[")
            and part.endswith(r"\]")
        ):

            latex = part[2:-2].strip()

            st.latex(
                latex
            )

        else:

            st.markdown(
                part
            )


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Industrial AI Copilot",
    page_icon="⚙️",
    layout="wide",
)

st.title(
    "⚙️ Industrial AI Copilot"
)

st.write(
    "Engineering document intelligence, simulation-data analysis, "
    "and AI-assisted digital twin monitoring."
)


# ============================================================
# SIDEBAR
# ============================================================

mode = st.sidebar.radio(
    "Select mode",
    [
        "Document RAG",
        "CSV Analysis",
        "Digital Twin",
    ],
)


# ============================================================
# DOCUMENT RAG
# ============================================================

if mode == "Document RAG":

    st.header(
        "📄 Document RAG"
    )

    st.write(
        "Upload a technical PDF and ask questions based on its content."
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_upload",
    )

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()

        pdf_path = (
            f"data/{file_hash}.pdf"
        )

        cache_path = (
            f"data/{file_hash}_embeddings.json"
        )

        if not os.path.exists(
            pdf_path
        ):

            with open(
                pdf_path,
                "wb",
            ) as file:

                file.write(
                    pdf_bytes
                )

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

        chunks, page_numbers = load_pdf_chunks(
            pdf_path
        )

        (
            chunks,
            page_numbers,
            embeddings,
        ) = load_or_create_embeddings(
            client=client,
            chunks=chunks,
            page_numbers=page_numbers,
            cache_path=cache_path,
        )

        st.caption(
            f"Pages: {len(set(page_numbers))} "
            f"| Chunks: {len(chunks)} "
            f"| Retrieval: Hybrid "
            f"| Top K: 6"
        )

        question = st.text_input(
            "Ask a question about this document:"
        )

        if st.button(
            "Ask",
            key="pdf_ask",
        ):

            if not question:

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching the document..."
                ):

                    top_results = retrieve_top_chunks(
                        client=client,
                        question=question,
                        chunks=chunks,
                        embeddings=embeddings,
                        top_k=6,
                    )

                    context_parts = []

                    for (
                        final_score,
                        semantic_score,
                        keyword_score,
                        index,
                    ) in top_results:

                        page = page_numbers[index]
                        text = chunks[index]

                        context_parts.append(
                            f"[Page {page}]\n{text}"
                        )

                    context = "\n\n".join(
                        context_parts
                    )

                    prompt = f"""
You are an engineering research assistant.

Answer the user's question using only the retrieved
document context below.

Requirements:

1. Ground factual claims in the retrieved context.
2. Cite relevant pages using [Page X].
3. Do not invent unsupported information.
4. If the context is insufficient, say so clearly.
5. Use block LaTeX for useful equations in this format:

\\[
equation
\\]

6. Explain engineering concepts clearly and technically.

Retrieved context:

{context}

Question:

{question}
"""

                    response = client.responses.create(
                        model="gpt-5.6-luna",
                        input=prompt,
                    )

                st.subheader(
                    "Answer"
                )

                render_answer_with_latex(
                    response.output_text
                )

                st.subheader(
                    "Sources"
                )

                for rank, (
                    final_score,
                    semantic_score,
                    keyword_score,
                    index,
                ) in enumerate(
                    top_results,
                    start=1,
                ):

                    st.write(
                        f"{rank}. "
                        f"Page {page_numbers[index]} "
                        f"| Chunk {index} "
                        f"| Final {final_score:.4f} "
                        f"| Semantic {semantic_score:.4f} "
                        f"| Keyword {keyword_score:.4f}"
                    )


# ============================================================
# CSV ANALYSIS
# ============================================================

elif mode == "CSV Analysis":

    st.header(
        "📊 Engineering Data Analysis"
    )

    st.write(
        "Upload simulation or experimental CSV data for "
        "numerical analysis and AI interpretation."
    )

    uploaded_csv = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        key="csv_upload",
    )

    if uploaded_csv is not None:

        df = load_csv(
            uploaded_csv
        )

        st.success(
            f"Loaded: {uploaded_csv.name}"
        )

        numeric_columns = get_numeric_columns(
            df
        )

        # ====================================================
        # DATASET OVERVIEW
        # ====================================================

        st.subheader(
            "Dataset Overview"
        )

        col1, col2, col3 = st.columns(
            3
        )

        col1.metric(
            "Rows",
            len(df),
        )

        col2.metric(
            "Columns",
            len(df.columns),
        )

        col3.metric(
            "Numeric columns",
            len(numeric_columns),
        )

        # ====================================================
        # DATA PREVIEW
        # ====================================================

        st.subheader(
            "Data Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True,
        )

        # ====================================================
        # STATISTICS
        # ====================================================

        st.subheader(
            "Statistical Summary"
        )

        summary = get_basic_summary(
            df
        )

        st.dataframe(
            summary,
            use_container_width=True,
        )

        if len(
            numeric_columns
        ) >= 2:

            # =================================================
            # CORRELATION MATRIX
            # =================================================

            st.subheader(
                "Correlation Matrix"
            )

            correlation_matrix = get_correlation_matrix(
                df
            )

            st.dataframe(
                correlation_matrix
                .style
                .format("{:.3f}"),
                use_container_width=True,
            )

            # =================================================
            # PLOT
            # =================================================

            st.subheader(
                "Plot Data"
            )

            x_column = st.selectbox(
                "X axis",
                numeric_columns,
                index=0,
                key="x_axis",
            )

            y_column = st.selectbox(
                "Y axis",
                numeric_columns,
                index=1,
                key="y_axis",
            )

            if st.button(
                "Plot",
                key="csv_plot",
            ):

                fig, ax = plt.subplots()

                ax.plot(
                    df[x_column],
                    df[y_column],
                )

                ax.set_xlabel(
                    x_column
                )

                ax.set_ylabel(
                    y_column
                )

                ax.set_title(
                    f"{y_column} vs {x_column}"
                )

                ax.grid(
                    True
                )

                st.pyplot(
                    fig
                )

                plt.close(
                    fig
                )

            # =================================================
            # TARGET CORRELATION
            # =================================================

            st.subheader(
                "Target Variable Analysis"
            )

            target_column = st.selectbox(
                "Select target variable",
                numeric_columns,
                key="target_column",
            )

            target_correlations = get_target_correlations(
                df,
                target_column,
            )

            if target_correlations is not None:

                st.write(
                    f"Correlation with **{target_column}**:"
                )

                st.dataframe(
                    target_correlations
                    .rename("correlation")
                    .to_frame()
                    .style
                    .format("{:.4f}"),
                    use_container_width=True,
                )

            # =================================================
            # LINEAR REGRESSION
            # =================================================

            st.subheader(
                "Regression Analysis"
            )

            regression_target = st.selectbox(
                "Regression target",
                numeric_columns,
                key="regression_target",
            )

            available_features = [
                column
                for column in numeric_columns
                if column != regression_target
            ]

            regression_features = st.multiselect(
                "Input features",
                available_features,
                default=available_features[:3],
                key="regression_features",
            )

            if st.button(
                "Run Regression",
                key="run_regression",
            ):

                if not regression_features:

                    st.warning(
                        "Please select at least one input feature."
                    )

                else:

                    regression = run_linear_regression(
                        df=df,
                        target_column=regression_target,
                        feature_columns=regression_features,
                    )

                    col1, col2, col3 = st.columns(
                        3
                    )

                    col1.metric(
                        "R²",
                        f"{regression['r2']:.4f}",
                    )

                    col2.metric(
                        "MSE",
                        f"{regression['mse']:.4f}",
                    )

                    col3.metric(
                        "Intercept",
                        f"{regression['intercept']:.4f}",
                    )

                    st.write(
                        "Regression coefficients:"
                    )

                    coefficient_df = (
                        regression[
                            "coefficients"
                        ]
                        .sort_values(
                            key=abs,
                            ascending=False,
                        )
                        .to_frame()
                    )

                    st.dataframe(
                        coefficient_df
                        .style
                        .format("{:.6f}"),
                        use_container_width=True,
                    )

                    regression_results = regression[
                        "results"
                    ]

                    st.write(
                        "Actual vs predicted:"
                    )

                    fig, ax = plt.subplots()

                    ax.scatter(
                        regression_results["actual"],
                        regression_results["predicted"],
                    )

                    minimum = min(
                        regression_results["actual"].min(),
                        regression_results["predicted"].min(),
                    )

                    maximum = max(
                        regression_results["actual"].max(),
                        regression_results["predicted"].max(),
                    )

                    ax.plot(
                        [minimum, maximum],
                        [minimum, maximum],
                    )

                    ax.set_xlabel(
                        "Actual"
                    )

                    ax.set_ylabel(
                        "Predicted"
                    )

                    ax.set_title(
                        f"Actual vs Predicted: "
                        f"{regression_target}"
                    )

                    ax.grid(
                        True
                    )

                    st.pyplot(
                        fig
                    )

                    plt.close(
                        fig
                    )

            # =================================================
            # AI DATA ANALYSIS
            # =================================================

            st.subheader(
                "AI Data Analysis"
            )

            data_question = st.text_input(
                "Ask a question about the dataset:",
                placeholder=(
                    "Example: Which parameter has the strongest "
                    "relationship with torque?"
                ),
                key="data_question",
            )

            if st.button(
                "Analyze",
                key="csv_analyze",
            ):

                if not data_question:

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Analyzing engineering data..."
                    ):

                        summary_text = (
                            summary.to_string()
                        )

                        correlation_text = (
                            correlation_matrix
                            .round(4)
                            .to_string()
                        )

                        if target_correlations is not None:

                            target_text = (
                                target_correlations
                                .round(4)
                                .to_string()
                            )

                        else:

                            target_text = (
                                "No target correlation data."
                            )

                        sample_text = (
                            df.head(20)
                            .to_string(
                                index=False
                            )
                        )

                        prompt = f"""
You are an engineering data analysis assistant.

Analyze the dataset using only the numerical
information provided below.

Important rules:

1. Distinguish correlation from causation.
2. Do not claim that a parameter physically causes
   an effect merely because correlation is high.
3. Explain positive and negative correlations clearly.
4. Mention possible nonlinear relationships if relevant.
5. If the available statistics are insufficient,
   say so clearly.
6. Focus on engineering interpretation.
7. Use units from the column names when available.
8. Consider possible multicollinearity between inputs.

User question:

{data_question}

Dataset sample:

{sample_text}

Statistical summary:

{summary_text}

Correlation matrix:

{correlation_text}

Selected target variable:

{target_column}

Target correlations:

{target_text}
"""

                        analysis_response = client.responses.create(
                            model="gpt-5.6-luna",
                            input=prompt,
                        )

                    st.subheader(
                        "AI Analysis"
                    )

                    st.markdown(
                        analysis_response.output_text
                    )

        else:

            st.warning(
                "The CSV needs at least two numeric "
                "columns for analysis."
            )


# ============================================================
# DIGITAL TWIN
# ============================================================

elif mode == "Digital Twin":

    st.header(
        "🧠 AI-Assisted Digital Twin"
    )

    st.write(
        "Train a PyTorch surrogate model using simulation or "
        "experimental data, then predict and monitor system "
        "behavior at new operating points."
    )

    twin_csv = st.file_uploader(
        "Upload training CSV",
        type=["csv"],
        key="digital_twin_csv",
    )

    if twin_csv is not None:

        df_twin = load_csv(
            twin_csv
        )

        st.success(
            f"Loaded: {twin_csv.name}"
        )

        numeric_columns_twin = get_numeric_columns(
            df_twin
        )

        # ====================================================
        # TRAINING DATA
        # ====================================================

        st.subheader(
            "Training Data"
        )

        st.dataframe(
            df_twin.head(20),
            use_container_width=True,
        )

        if len(
            numeric_columns_twin
        ) < 2:

            st.warning(
                "The dataset needs at least two numeric columns."
            )

        else:

            # =================================================
            # 1. CONFIGURE MODEL
            # =================================================

            st.subheader(
                "1. Configure Surrogate Model"
            )

            target_twin = st.selectbox(
                "Prediction target",
                numeric_columns_twin,
                key="twin_target",
            )

            available_features_twin = [
                column
                for column in numeric_columns_twin
                if column != target_twin
            ]

            features_twin = st.multiselect(
                "Input parameters",
                available_features_twin,
                default=available_features_twin[:3],
                key="twin_features",
            )

            epochs = st.slider(
                "Training epochs",
                min_value=100,
                max_value=3000,
                value=1000,
                step=100,
                key="twin_epochs",
            )

            learning_rate = st.selectbox(
                "Learning rate",
                [
                    0.001,
                    0.005,
                    0.01,
                ],
                index=2,
                key="twin_lr",
            )

            # =================================================
            # TRAIN MODEL
            # =================================================

            if st.button(
                "Train Digital Twin Model",
                key="train_twin",
            ):

                if not features_twin:

                    st.warning(
                        "Please select at least one input parameter."
                    )

                else:

                    with st.spinner(
                        "Training PyTorch surrogate model..."
                    ):

                        twin_result = train_surrogate_model(
                            df=df_twin,
                            feature_columns=features_twin,
                            target_column=target_twin,
                            epochs=epochs,
                            learning_rate=learning_rate,
                        )

                    st.session_state[
                        "twin_result"
                    ] = twin_result

                    st.session_state.pop(
                        "twin_prediction",
                        None,
                    )

                    st.session_state.pop(
                        "twin_ood",
                        None,
                    )

                    st.session_state.pop(
                        "actual_twin_value",
                        None,
                    )

                    st.success(
                        "Digital twin model trained successfully."
                    )

            # =================================================
            # MODEL RESULTS
            # =================================================

            if "twin_result" in st.session_state:

                twin_result = st.session_state[
                    "twin_result"
                ]

                # =================================================
                # 2. MODEL PERFORMANCE
                # =================================================

                st.subheader(
                    "2. Model Performance"
                )

                col1, col2, col3, col4 = st.columns(
                    4
                )

                col1.metric(
                    "Train R²",
                    f"{twin_result['train_r2']:.4f}",
                )

                col2.metric(
                    "Test R²",
                    f"{twin_result['test_r2']:.4f}",
                )

                col3.metric(
                    "Train MSE",
                    f"{twin_result['train_mse']:.6f}",
                )

                col4.metric(
                    "Test MSE",
                    f"{twin_result['test_mse']:.6f}",
                )

                r2_gap = (
                    twin_result["train_r2"]
                    - twin_result["test_r2"]
                )

                if twin_result[
                    "test_r2"
                ] < 0:

                    st.error(
                        "🔴 Poor generalization: "
                        "the model performs poorly "
                        "on unseen test data."
                    )

                elif r2_gap > 0.20:

                    st.warning(
                        "⚠️ Possible overfitting: "
                        "training performance is substantially "
                        "better than test performance."
                    )

                else:

                    st.success(
                        "🟢 Model generalization looks acceptable."
                    )

                # =================================================
                # TRAINING LOSS
                # =================================================

                st.write(
                    "Training loss:"
                )

                fig_loss, ax_loss = plt.subplots()

                ax_loss.plot(
                    twin_result["losses"]
                )

                ax_loss.set_xlabel(
                    "Epoch"
                )

                ax_loss.set_ylabel(
                    "MSE Loss"
                )

                ax_loss.set_title(
                    "PyTorch Training Loss"
                )

                ax_loss.grid(
                    True
                )

                st.pyplot(
                    fig_loss
                )

                plt.close(
                    fig_loss
                )

                # =================================================
                # TEST SET PERFORMANCE
                # =================================================

                st.write(
                    "Test data: Actual vs Predicted"
                )

                fig_pred, ax_pred = plt.subplots()

                ax_pred.scatter(
                    twin_result["test_actual"],
                    twin_result["test_predicted"],
                )

                minimum = min(
                    twin_result[
                        "test_actual"
                    ].min(),
                    twin_result[
                        "test_predicted"
                    ].min(),
                )

                maximum = max(
                    twin_result[
                        "test_actual"
                    ].max(),
                    twin_result[
                        "test_predicted"
                    ].max(),
                )

                ax_pred.plot(
                    [minimum, maximum],
                    [minimum, maximum],
                )

                ax_pred.set_xlabel(
                    "Actual"
                )

                ax_pred.set_ylabel(
                    "Predicted"
                )

                ax_pred.set_title(
                    f"Test Set Prediction: "
                    f"{twin_result['target_column']}"
                )

                ax_pred.grid(
                    True
                )

                st.pyplot(
                    fig_pred
                )

                plt.close(
                    fig_pred
                )

                # =================================================
                # 3. VIRTUAL OPERATING POINT
                # =================================================

                st.subheader(
                    "3. Virtual Operating Point"
                )

                st.write(
                    "Enter a new operating condition and let the "
                    "PyTorch surrogate model predict the system response."
                )

                input_values = []

                for feature in twin_result[
                    "feature_columns"
                ]:

                    feature_min = float(
                        df_twin[
                            feature
                        ].min()
                    )

                    feature_max = float(
                        df_twin[
                            feature
                        ].max()
                    )

                    feature_mean = float(
                        df_twin[
                            feature
                        ].mean()
                    )

                    value = st.number_input(
                        feature,
                        min_value=feature_min,
                        max_value=feature_max,
                        value=feature_mean,
                        key=f"input_{feature}",
                    )

                    input_values.append(
                        value
                    )

                # =================================================
                # PREDICT
                # =================================================

                if st.button(
                    "Predict System State",
                    key="predict_twin",
                ):

                    # ---------------------------------------------
                    # OOD CHECK
                    # ---------------------------------------------

                    out_of_range = check_out_of_distribution(
                        twin_result,
                        input_values,
                    )

                    st.session_state[
                        "twin_ood"
                    ] = out_of_range

                    # ---------------------------------------------
                    # IMPORTANT:
                    # Prediction happens whether OOD or NOT
                    # ---------------------------------------------

                    prediction = predict_surrogate(
                        twin_result,
                        input_values,
                    )

                    st.session_state[
                        "twin_prediction"
                    ] = prediction

                    # Reset measured value to latest prediction
                    st.session_state[
                        "actual_twin_value"
                    ] = float(
                        prediction
                    )

                # =================================================
                # OOD STATUS
                # =================================================

                if "twin_ood" in st.session_state:

                    out_of_range = st.session_state[
                        "twin_ood"
                    ]

                    if out_of_range:

                        st.warning(
                            "⚠️ Out-of-distribution input detected. "
                            "The operating point is outside the "
                            "training-data range."
                        )

                        for item in out_of_range:

                            st.write(
                                f"- {item['feature']}: "
                                f"{item['value']:.4f} "
                                f"(training range: "
                                f"{item['min']:.4f} – "
                                f"{item['max']:.4f})"
                            )

                    else:

                        st.success(
                            "🟢 Operating point is inside "
                            "the training-data range."
                        )

                # =================================================
                # PREDICTION RESULT
                # =================================================

                if "twin_prediction" in st.session_state:

                    prediction = st.session_state[
                        "twin_prediction"
                    ]

                    st.subheader(
                        "Digital Twin Prediction"
                    )

                    st.metric(
                        twin_result[
                            "target_column"
                        ],
                        f"{prediction:.4f}",
                    )

                    # =================================================
                    # 4. STATE MONITORING
                    # =================================================

                    st.subheader(
                        "4. State Monitoring"
                    )

                    st.write(
                        "Enter the actual measured value from the "
                        "physical or simulated system."
                    )

                    actual_value = st.number_input(
                        (
                            f"Actual measured "
                            f"{twin_result['target_column']}"
                        ),
                        key="actual_twin_value",
                    )

                    residual = (
                        actual_value
                        - prediction
                    )

                    absolute_error = abs(
                        residual
                    )

                    if abs(
                        prediction
                    ) > 1e-8:

                        percentage_error = (
                            absolute_error
                            / abs(prediction)
                            * 100
                        )

                    else:

                        percentage_error = 0.0

                    # =================================================
                    # RESIDUAL THRESHOLD
                    # =================================================

                    training_residuals = twin_result[
                        "training_residuals"
                    ]

                    residual_std = float(
                        training_residuals.std()
                    )

                    warning_threshold = (
                        2
                        * residual_std
                    )

                    alarm_threshold = (
                        3
                        * residual_std
                    )

                    # =================================================
                    # MONITORING METRICS
                    # =================================================

                    col1, col2, col3 = st.columns(
                        3
                    )

                    col1.metric(
                        "Predicted",
                        f"{prediction:.4f}",
                    )

                    col2.metric(
                        "Actual",
                        f"{actual_value:.4f}",
                    )

                    col3.metric(
                        "Deviation",
                        f"{percentage_error:.2f}%",
                    )

                    st.write(
                        f"Residual: **{residual:.4f}**"
                    )

                    st.write(
                        f"Warning threshold: "
                        f"±{warning_threshold:.4f}"
                    )

                    st.write(
                        f"Alarm threshold: "
                        f"±{alarm_threshold:.4f}"
                    )

                    # =================================================
                    # STATE CLASSIFICATION
                    # =================================================

                    if absolute_error >= alarm_threshold:

                        system_status = (
                            "ALARM"
                        )

                        st.error(
                            "🔴 ALARM — The measured system state "
                            "deviates strongly from the digital "
                            "twin prediction."
                        )

                    elif absolute_error >= warning_threshold:

                        system_status = (
                            "WARNING"
                        )

                        st.warning(
                            "🟠 WARNING — The measured system state "
                            "shows abnormal deviation from the "
                            "digital twin."
                        )

                    else:

                        system_status = (
                            "NORMAL"
                        )

                        st.success(
                            "🟢 NORMAL — The measured system state "
                            "is consistent with the digital "
                            "twin prediction."
                        )

                    # =================================================
                    # 5. AI DIAGNOSIS
                    # =================================================

                    st.subheader(
                        "5. AI Diagnosis"
                    )

                    if st.button(
                        "Explain System State",
                        key="explain_twin_state",
                    ):

                        operating_points = []

                        for feature, value in zip(
                            twin_result[
                                "feature_columns"
                            ],
                            input_values,
                        ):

                            operating_points.append(
                                f"{feature}: {value}"
                            )

                        operating_text = "\n".join(
                            operating_points
                        )

                        diagnosis_prompt = f"""
You are an industrial engineering and digital twin assistant.

Analyze the current digital twin monitoring result.

Do not invent specific mechanical failure causes as facts.

Distinguish between observations, possible causes,
and recommended checks.

Prediction target:

{twin_result["target_column"]}

Operating point:

{operating_text}

Digital twin prediction:

{prediction:.6f}

Actual measured value:

{actual_value:.6f}

Residual:

{residual:.6f}

Percentage deviation:

{percentage_error:.2f}%

Warning threshold:

{warning_threshold:.6f}

Alarm threshold:

{alarm_threshold:.6f}

System status:

{system_status}

Explain:

1. What the deviation means.
2. Whether the state is normal, warning, or alarm.
3. Plausible engineering reasons to investigate.
4. What measurements or checks should be performed next.
5. Mention limitations of the current surrogate model.
"""

                        with st.spinner(
                            "Generating engineering diagnosis..."
                        ):

                            diagnosis_response = (
                                client.responses.create(
                                    model="gpt-5.6-luna",
                                    input=diagnosis_prompt,
                                )
                            )

                        st.markdown(
                            diagnosis_response.output_text
                        )