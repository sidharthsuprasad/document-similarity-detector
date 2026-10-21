import streamlit as st
import pandas as pd
from document_processor import extract_text, normalize_text, file_fingerprint
from similarity_engine import compare_documents, find_exact_duplicates
from report_generator import results_to_csv

st.set_page_config(page_title="AI Document Similarity & Duplicate Detector", page_icon="📄", layout="wide")

st.title("📄 AI Document Similarity & Duplicate Detector")
st.caption("Compare PDF, DOCX, and TXT documents using text processing and TF-IDF cosine similarity.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Near-duplicate threshold (%)", min_value=50, max_value=100, value=85, step=1)
    st.info("Similarity is a screening signal, not proof of plagiarism. Results depend on the selected text representation.")

uploaded_files = st.file_uploader(
    "Upload two or more documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="For best results, use text-based PDFs. Scanned image PDFs may require OCR, which is not included in this version."
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Upload at least two documents to compare them.")
        st.stop()

    documents = []
    extraction_errors = []
    seen_names = set()

    for uploaded in uploaded_files:
        name = uploaded.name
        if name in seen_names:
            name = f"{name} (uploaded copy {len(seen_names) + 1})"
        seen_names.add(uploaded.name)
        raw = uploaded.getvalue()
        try:
            text = extract_text(name, raw)
            cleaned = normalize_text(text)
            if not cleaned.strip():
                extraction_errors.append(f"{name}: no readable text was found.")
                continue
            documents.append({
                "name": name,
                "text": cleaned,
                "fingerprint": file_fingerprint(raw),
                "size_bytes": len(raw),
            })
        except Exception as exc:
            extraction_errors.append(f"{name}: {exc}")

    for error in extraction_errors:
        st.warning(error)

    if len(documents) < 2:
        st.error("At least two documents with readable text are needed. Check the file types and contents.")
        st.stop()

    st.subheader("Uploaded documents")
    summary = pd.DataFrame([
        {"Document": d["name"], "Words": len(d["text"].split()), "Size (KB)": round(d["size_bytes"] / 1024, 1)}
        for d in documents
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if st.button("Analyze documents", type="primary", use_container_width=True):
        with st.spinner("Extracting features and comparing document pairs..."):
            results, matrix = compare_documents(documents)
            exact_pairs = find_exact_duplicates(documents)

        st.session_state["results"] = results
        st.session_state["exact_pairs"] = exact_pairs
        st.session_state["doc_names"] = [d["name"] for d in documents]
        st.session_state["matrix"] = matrix
        st.session_state["threshold"] = threshold

if "results" in st.session_state:
    results = st.session_state["results"]
    exact_pairs = st.session_state["exact_pairs"]
    threshold = st.session_state.get("threshold", threshold)

    st.divider()
    st.subheader("Analysis results")
    total_pairs = len(results)
    near_pairs = sum(r["Similarity (%)"] >= threshold for r in results)
    exact_count = len(exact_pairs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents compared", len(st.session_state["doc_names"]))
    c2.metric("Document pairs", total_pairs)
    c3.metric("Pairs at/above threshold", near_pairs)

    if exact_pairs:
        st.success(f"Exact text duplicates found: {len(exact_pairs)} pair(s).")
        for a, b in exact_pairs:
            st.write(f"• **{a}** and **{b}** have identical normalized text.")
    else:
        st.info("No exact text duplicates were found.")

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["Classification"] = result_df["Similarity (%)"].apply(
            lambda score: "Near duplicate" if score >= threshold else "Review"
        )
        st.dataframe(
            result_df.sort_values("Similarity (%)", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Similarity (%)": st.column_config.ProgressColumn(
                    "Similarity (%)", min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )
        csv_bytes = results_to_csv(results).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="document_similarity_results.csv",
            mime="text/csv",
        )

        st.subheader("Compare a document pair")
        names = st.session_state["doc_names"]
        pair_options = [f"{r['Document A']}  ↔  {r['Document B']}" for r in results]
        selected = st.selectbox("Choose a pair", pair_options)
        selected_row = results[pair_options.index(selected)]
        st.metric("Pair similarity", f"{selected_row['Similarity (%)']:.2f}%")
        st.caption("This score reflects TF-IDF word-weight similarity. It does not establish authorship or intent.")

st.divider()
st.caption("Privacy note: this demo processes uploaded file contents in the current app session and does not intentionally save the source documents to a database. Avoid uploading confidential documents to public deployments.")
