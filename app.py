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
    run_linear_regression
)


# ============================================================
# ENVIRONMENT / OPENAI
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY was not found. "
        "Please check your .env file."
    )
    st.stop()


client = OpenAI(
    api_key=api_key
)
os.makedirs("data", exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_answer_with_latex(text):
    """
    Render normal Markdown text and block LaTeX equations.

    Expected block equation format:

    \\[
    equation
    \\]
    """

    parts = re.split(
        r"(\\\[[\s\S]*?\\\])",
        text
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
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Engineering AI Copilot",
    page_icon="⚙️",
    layout="wide"
)


st.title(
    "⚙️ Engineering AI Copilot"
)

st.write(
    "Analyze engineering documents and simulation data with AI."
)


# ============================================================
# SIDEBAR
# ============================================================

mode = st.sidebar.radio(
    "Select mode",
    [
        "Document RAG",
        "CSV Analysis"
    ]
)


# ============================================================
# DOCUMENT RAG MODE
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
        key="pdf_upload"
    )


    if uploaded_file is not None:

        # ----------------------------------------------------
        # Read uploaded PDF bytes
        # ----------------------------------------------------

        pdf_bytes = uploaded_file.getvalue()


        # ----------------------------------------------------
        # Generate unique file ID
        # ----------------------------------------------------

        file_hash = hashlib.md5(
            pdf_bytes
        ).hexdigest()


        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        pdf_path = (
            f"data/{file_hash}.pdf"
        )

        cache_path = (
            f"data/{file_hash}_embeddings.json"
        )


        # ----------------------------------------------------
        # Save uploaded PDF
        # ----------------------------------------------------

        if not os.path.exists(
            pdf_path
        ):

            with open(
                pdf_path,
                "wb"
            ) as file:

                file.write(
                    pdf_bytes
                )


        st.success(
            f"Loaded: {uploaded_file.name}"
        )


        # ----------------------------------------------------
        # Read PDF and split into chunks
        # ----------------------------------------------------

        chunks, page_numbers = load_pdf_chunks(
            pdf_path
        )


        # ----------------------------------------------------
        # Load or create embeddings
        # ----------------------------------------------------

        (
            chunks,
            page_numbers,
            embeddings
        ) = load_or_create_embeddings(
            client=client,
            chunks=chunks,
            page_numbers=page_numbers,
            cache_path=cache_path
        )


        # ----------------------------------------------------
        # Document information
        # ----------------------------------------------------

        st.caption(
            f"Pages: {len(set(page_numbers))} "
            f"| Chunks: {len(chunks)} "
            f"| Retrieval: Hybrid "
            f"| Top K: 6"
        )


        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        question = st.text_input(
            "Ask a question about this document:"
        )


        if st.button(
            "Ask",
            key="pdf_ask"
        ):

            if not question:

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching the document..."
                ):

                    # -----------------------------------------
                    # Hybrid retrieval
                    # -----------------------------------------

                    top_results = retrieve_top_chunks(
                        client=client,
                        question=question,
                        chunks=chunks,
                        embeddings=embeddings,
                        top_k=6
                    )


                    # -----------------------------------------
                    # Build context
                    # -----------------------------------------

                    context_parts = []

                    for (
                        final_score,
                        semantic_score,
                        keyword_score,
                        index
                    ) in top_results:

                        page = (
                            page_numbers[index]
                        )

                        text = (
                            chunks[index]
                        )

                        context_parts.append(
                            f"[Page {page}]\n{text}"
                        )


                    context = "\n\n".join(
                        context_parts
                    )


                    # -----------------------------------------
                    # Prompt
                    # -----------------------------------------

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


                    # -----------------------------------------
                    # Generate answer
                    # -----------------------------------------

                    response = client.responses.create(
                        model="gpt-5.6-luna",
                        input=prompt
                    )


                # ---------------------------------------------
                # Answer
                # ---------------------------------------------

                st.subheader(
                    "Answer"
                )

                render_answer_with_latex(
                    response.output_text
                )


                # ---------------------------------------------
                # Sources
                # ---------------------------------------------

                st.subheader(
                    "Sources"
                )

                for rank, (
                    final_score,
                    semantic_score,
                    keyword_score,
                    index
                ) in enumerate(
                    top_results,
                    start=1
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
# CSV ANALYSIS MODE
# ============================================================

elif mode == "CSV Analysis":

    st.header(
        "📊 CSV Engineering Data Analysis"
    )

    st.write(
        "Upload simulation or experimental CSV data "
        "for numerical analysis and AI interpretation."
    )


    uploaded_csv = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        key="csv_upload"
    )


    if uploaded_csv is not None:

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

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
            len(df)
        )


        col2.metric(
            "Columns",
            len(df.columns)
        )


        col3.metric(
            "Numeric columns",
            len(numeric_columns)
        )


        # ====================================================
        # DATA PREVIEW
        # ====================================================

        st.subheader(
            "Data Preview"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


        # ====================================================
        # STATISTICAL SUMMARY
        # ====================================================

        st.subheader(
            "Statistical Summary"
        )


        summary = get_basic_summary(
            df
        )


        st.dataframe(
            summary,
            use_container_width=True
        )


        # ====================================================
        # CHECK NUMERIC DATA
        # ====================================================

        if len(numeric_columns) >= 2:

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
                use_container_width=True
            )


            # =================================================
            # PLOT DATA
            # =================================================

            st.subheader(
                "Plot Data"
            )


            x_column = st.selectbox(
                "X axis",
                numeric_columns,
                index=0,
                key="x_axis"
            )


            y_column = st.selectbox(
                "Y axis",
                numeric_columns,
                index=1,
                key="y_axis"
            )


            if st.button(
                "Plot",
                key="csv_plot"
            ):

                fig, ax = plt.subplots()


                ax.plot(
                    df[x_column],
                    df[y_column]
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


            # =================================================
            # TARGET VARIABLE ANALYSIS
            # =================================================

            st.subheader(
                "Target Variable Analysis"
            )


            target_column = st.selectbox(
                "Select target variable",
                numeric_columns,
                key="target_column"
            )


            target_correlations = get_target_correlations(
                df,
                target_column
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
                    use_container_width=True
                )


            # =================================================
            # REGRESSION ANALYSIS
            # =================================================

            st.subheader(
                "Regression Analysis"
            )


            regression_target = st.selectbox(
                "Regression target",
                numeric_columns,
                key="regression_target"
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
                key="regression_features"
            )


            if st.button(
                "Run Regression",
                key="run_regression"
            ):

                if not regression_features:

                    st.warning(
                        "Please select at least one input feature."
                    )

                else:

                    regression = run_linear_regression(
                        df=df,
                        target_column=regression_target,
                        feature_columns=regression_features
                    )


                    # -----------------------------------------
                    # Regression metrics
                    # -----------------------------------------

                    col1, col2, col3 = st.columns(
                        3
                    )


                    col1.metric(
                        "R²",
                        f"{regression['r2']:.4f}"
                    )


                    col2.metric(
                        "MSE",
                        f"{regression['mse']:.4f}"
                    )


                    col3.metric(
                        "Intercept",
                        f"{regression['intercept']:.4f}"
                    )


                    # -----------------------------------------
                    # Regression coefficients
                    # -----------------------------------------

                    st.write(
                        "Regression coefficients:"
                    )


                    coefficient_df = (
                        regression["coefficients"]
                        .sort_values(
                            key=abs,
                            ascending=False
                        )
                        .to_frame()
                    )


                    st.dataframe(
                        coefficient_df
                        .style
                        .format("{:.6f}"),
                        use_container_width=True
                    )


                    # -----------------------------------------
                    # Actual vs predicted
                    # -----------------------------------------

                    st.write(
                        "Actual vs. predicted:"
                    )


                    regression_results = (
                        regression["results"]
                    )


                    fig, ax = plt.subplots()


                    ax.scatter(
                        regression_results["actual"],
                        regression_results["predicted"]
                    )


                    minimum = min(
                        regression_results["actual"].min(),
                        regression_results["predicted"].min()
                    )


                    maximum = max(
                        regression_results["actual"].max(),
                        regression_results["predicted"].max()
                    )


                    ax.plot(
                        [minimum, maximum],
                        [minimum, maximum]
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
                key="data_question"
            )


            if st.button(
                "Analyze",
                key="csv_analyze"
            ):

                if not data_question:

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Analyzing engineering data..."
                    ):

                        # -------------------------------------
                        # Prepare dataset information
                        # -------------------------------------

                        summary_text = (
                            summary
                            .to_string()
                        )


                        correlation_text = (
                            correlation_matrix
                            .round(4)
                            .to_string()
                        )


                        target_text = (
                            target_correlations
                            .round(4)
                            .to_string()
                        )


                        sample_text = (
                            df.head(20)
                            .to_string(
                                index=False
                            )
                        )


                        # -------------------------------------
                        # Prompt
                        # -------------------------------------

                        prompt = f"""
You are an engineering data analysis assistant.

Analyze the dataset only using the numerical
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


                        # -------------------------------------
                        # AI response
                        # -------------------------------------

                        analysis_response = (
                            client.responses.create(
                                model="gpt-5.6-luna",
                                input=prompt
                            )
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