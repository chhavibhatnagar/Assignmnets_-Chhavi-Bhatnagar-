import streamlit as st
from rag_pipeline import (
    split_documents,
    get_embedding_model,
    build_vectorstore,
    build_vectorstore_chroma,
    retrieve,
    get_llm,
    generate_answer,
    get_document_headings,
    extract_headings,
)
from pdf_highlighter import highlight_pdf
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import sys
import types
import re
import math

fake_module = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:
    pass


fake_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = fake_module


from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
   

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚")


def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if (username == st.secrets["credentials"]["username"] and
                    password == st.secrets["credentials"]["password"]):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.stop()


check_auth()

st.title("📚 Smart Research Assistant")
st.caption("Upload documents and ask questions — answers are grounded in retrieved context. Type 'exit' to end the conversation.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "conversation_ended" not in st.session_state:
    st.session_state["conversation_ended"] = False
if "preset_query" not in st.session_state:
    st.session_state["preset_query"] = None


@st.cache_resource
def get_embedding_model_cached(model_name):
    return get_embedding_model(model_name)


@st.cache_resource
def get_llm_cached():
    return get_llm()


@st.cache_resource
def get_judge_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def generate_reference_answer(query, retrieved_docs, judge_llm):
    """Generate a reference answer for RAGAS Answer Correctness."""
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
Using ONLY the context below, generate the ideal reference answer.

Context:
{context}

Question:
{query}
"""

    return judge_llm.invoke(prompt).content

def score_answer(query, answer, retrieved_docs, judge_llm):
    """Run RAGAS faithfulness + answer relevancy. Truncates answer for scoring stability."""
    dataset = Dataset.from_dict({
        "question": [query],
        "answer": [answer[:800]],  
        "contexts": [[doc.page_content for doc in retrieved_docs]],
    })
    result = ragas_evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=judge_llm,
    )
    return result

def render_source_evidence(results, query, max_per_source=5):
    st.markdown("**Source Evidence**")
    stop_words = {
        "what", "is", "the", "a", "an", "of", "to",
        "in", "on", "for", "how", "why", "when", "where", "which"
    }
    keywords = [k for k in re.findall(r"\w+", query.lower()) if k not in stop_words]

    source_counts = {}
    for rank, doc in enumerate(results, start=1):
        source_name = doc.metadata.get("source", "Unknown")
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        if source_counts[source_name] > max_per_source:
            continue


        page_number = doc.metadata.get("page", 0)

        snippet = doc.page_content
        if keywords:
            first = keywords[0]
            match = re.search(re.escape(first), snippet, re.IGNORECASE)
            if match:
                start = max(0, match.start() - 120)
                end = min(len(snippet), match.end() + 180)
                snippet = snippet[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(doc.page_content):
                    snippet += "..."
        for word in keywords:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

        with st.container(border=True):
            st.markdown(snippet, unsafe_allow_html=True)
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 2])
            col1.metric("Page", int(page_number) + 1)
            col2.metric("Relevance Rank", rank)
            col3.write("**Source**")
            col3.write(source_name.replace("_", " ").replace(".pdf", ""))


def process_query(query, vectorstore, llm, selected_language, selected_domain):
    """Shared logic for processing any query — typed or preset button."""
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        results = []
        highlight_files = {}

        if query == "__HIGHLIGHT__":
            import fitz
            pdf_paths = st.session_state.get("pdf_paths")
            if not pdf_paths:
                st.error("PDF file(s) not found. Please re-upload your documents.")
                return

            answer = "**📑 Important Sections Found**\n\n"
            with st.spinner("Extracting important headings and highlighting..."):
                for filename, pdf_path in pdf_paths.items():
                    if not os.path.exists(pdf_path):
                        continue

                    doc = fitz.open(pdf_path)
                    page_texts = {i: page.get_text() for i, page in enumerate(doc)}
                    doc.close()

                    raw_headings = extract_headings(page_texts)
                    headings = [h["heading"] for h in raw_headings if "heading" in h]

                    if not headings:
                        answer += f"**{filename}**: no headings found\n\n"
                        continue

                    output_name = f"highlighted_{filename}"
                    matched_count = highlight_pdf(pdf_path, headings, output_name)

                    answer += f"**{filename}** ({matched_count}/{len(headings)} highlighted)\n\n"
                    for heading in headings:
                        if heading.strip():
                            answer += f"✅ {heading.strip()}\n"
                    answer += "\n"

                    with open(output_name, "rb") as f:
                        highlight_files[filename] = f.read()

            st.write(answer)


            turn_index = len(st.session_state["chat_history"])
            st.session_state["chat_history"].append({
                "question": "🟨 Highlight PDF",
                "answer": answer,
                "results": [],
                "faithfulness": None,
                "relevancy": None,
                "highlight_files": highlight_files,
            })

            for filename, pdf_bytes in highlight_files.items():
                st.download_button(
                    f"📥 Download Highlighted {filename}",
                    pdf_bytes,
                    file_name=f"highlighted_{filename}",
                    mime="application/pdf",
                    key=f"download_{filename}_{turn_index}",
                )
            return

        with st.spinner("Searching documents and generating answer..."):
            results = retrieve(query, vectorstore)
            answer = generate_answer(
                query, results, llm,
                st.session_state["chat_history"],
                selected_language,
                selected_domain,
            )
            OUT_OF_SCOPE_MESSAGE = (
                "I couldn't find information about this in the uploaded document(s). "
                "Please ask a question related to the uploaded document."
                )
            
            is_out_of_scope = answer.strip() == OUT_OF_SCOPE_MESSAGE

        st.write(answer)


        turn_index = len(st.session_state["chat_history"])
        st.session_state["chat_history"].append({
            "question": query,
            "answer": answer,
            "results": results,
            "faithfulness": None,
            "relevancy": None,
            "highlight_files": {},
        })


        OUT_OF_SCOPE_MESSAGE = (
            "I couldn't find information about this in the uploaded document(s). "
            "Please ask a question related to the uploaded document."
        )
        if "I couldn't find information about this" in answer:
            col1, col2, col3 = st.columns(3)
            col1.metric("Context Alignment", "--")
            col2.metric("Answer Correctness", "--")
            col3.metric("Overall Quality", "--")
        else:
            metrics_placeholder = st.empty()
            metrics_placeholder.info("Scoring answer reliability...")
            try:
                judge_llm = get_judge_llm()
                f, r = float("nan"), float("nan")
                
                for attempt in range(2):
                    eval_result = score_answer(query, answer, results, judge_llm)
                    f = eval_result["faithfulness"][0]
                    r = eval_result["answer_relevancy"][0]
                    if not (math.isnan(f) or math.isnan(r)):
                        break
                
                overall_quality = (f + r) / 2
                metrics_placeholder.empty()
                if not math.isnan(f):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Context Alignment", f"{f:.2f}")
                    col2.metric("Answer Correctness", f"{r:.2f}")
                    col3.metric("Overall Quality", f"{overall_quality:.2f}")
                    
                    st.session_state["chat_history"][turn_index]["faithfulness"] = f
                    st.session_state["chat_history"][turn_index]["relevancy"] = r
                    st.session_state["chat_history"][turn_index]["overall_quality"] = overall_quality
                
                else:
                    st.warning("Could not score this response — try asking again.")
                    
            except Exception as e:
                metrics_placeholder.empty()
                st.error(f"Evaluation failed: {e}")

        if not is_out_of_scope:
            with st.expander("Show source evidence"):
                render_source_evidence(results, query)


EMBEDDING_OPTIONS = {
    "MiniLM (fast, good general quality)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (slower, higher quality)": "sentence-transformers/all-mpnet-base-v2",
}
selected_embedding_label = st.sidebar.selectbox("Embedding model", options=list(EMBEDDING_OPTIONS.keys()))
selected_embedding_name = EMBEDDING_OPTIONS[selected_embedding_label]

LANGUAGE_OPTIONS = [
    "Auto-detect", "English", "Hindi", "Spanish",
    "French", "German", "Chinese (Simplified)", "Arabic",
]
selected_language = st.sidebar.selectbox("Response language", options=LANGUAGE_OPTIONS)

DOMAIN_OPTIONS = ["General", "Legal", "Medical", "Finance", "Research"]
selected_domain = st.sidebar.selectbox("Assistant domain", options=DOMAIN_OPTIONS)

VECTORSTORE_OPTIONS = ["FAISS", "ChromaDB"]
selected_vectorstore = st.sidebar.selectbox("Vector store", options=VECTORSTORE_OPTIONS)

if st.session_state.get("last_vectorstore_type") != selected_vectorstore:
    st.session_state.pop("vectorstore", None)
    st.session_state["last_vectorstore_type"] = selected_vectorstore

if st.session_state.get("last_embedding_model") != selected_embedding_name:
    st.session_state.pop("vectorstore", None)
    st.session_state["last_embedding_model"] = selected_embedding_name

embedding_model = get_embedding_model_cached(selected_embedding_name)
llm = get_llm_cached()

uploaded_files = st.sidebar.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Clear conversation"):
    st.session_state["chat_history"] = []
    st.session_state["conversation_ended"] = False
    st.session_state["preset_query"] = None
    st.rerun()

if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.rerun()


if uploaded_files:
    if "vectorstore" not in st.session_state or st.session_state.get("uploaded_names") != [f.name for f in uploaded_files]:
        with st.spinner("Processing documents..."):
            all_docs = []
            pdf_paths_by_name = {}
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                os.makedirs("uploaded_pdfs", exist_ok=True)
                saved_path = os.path.join("uploaded_pdfs", uploaded_file.name)
                with open(saved_path, "wb") as f:
                    f.write(open(tmp_path, "rb").read())
                pdf_paths_by_name[uploaded_file.name] = saved_path

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["filepath"] = saved_path
                all_docs.extend(docs)

            chunks = split_documents(all_docs)

            if selected_vectorstore == "ChromaDB":
                vectorstore = build_vectorstore_chroma(chunks, embedding_model, selected_embedding_name)
            else:
                vectorstore = build_vectorstore(chunks, embedding_model)

            st.session_state["vectorstore"] = vectorstore
            st.session_state["uploaded_names"] = [f.name for f in uploaded_files]
            st.session_state["pdf_paths"] = pdf_paths_by_name

        st.sidebar.success(f"Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks.")


if "vectorstore" not in st.session_state:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for turn_index, turn in enumerate(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("faithfulness") is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("Faithfulness", f"{turn['faithfulness']:.2f}")
                col2.metric("Answer Relevancy", f"{turn['relevancy']:.2f}")
                col3.metric("Overall Quality", f"{turn['overall_quality']:.2f}")
            if (
                turn.get("results")
                and "I couldn't find information about this" not in turn["answer"]
            ):
                with st.expander("Show source evidence"):
                    render_source_evidence(turn["results"], turn["question"])
            if turn.get("highlight_files"):
                for filename, pdf_bytes in turn["highlight_files"].items():
                    st.download_button(
                        f"📥 Download Highlighted {filename}",
                        pdf_bytes,
                        file_name=f"highlighted_{filename}",
                        mime="application/pdf",
                        key=f"replay_download_{turn_index}_{filename}",
                    )

    if st.session_state["conversation_ended"]:
        st.info("Conversation ended. Click 'Clear conversation' in the sidebar to start a new one.")
    else:
        st.markdown("**Quick Actions**")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        _, center, _ = st.columns([1, 2, 1])

        PRESET_QUERIES = {
            "📄 Summarize": "Give a comprehensive, detailed summary of this document covering all major sections and topics.",
            "🎯 Key Points": "What are the key points and most important ideas in this document? List them clearly.",
            "💡 Future Work": "What future work, recommendations, or next steps does this document suggest?",
            "❓ Viva Questions": "Generate 5 important exam or viva questions based on this document, with detailed answers for each.",
            "📊 Explain Simply": "Explain the main concepts of this document in very simple language, as if explaining to a 10-year-old.",
            "🔍 Research Gaps": "What are the limitations, gaps, or unanswered questions identified in this document?",
        }

        if col1.button("📄 Summarize", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["📄 Summarize"]
            st.rerun()
        if col2.button("🎯 Key Points", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["🎯 Key Points"]
            st.rerun()
        if col3.button("💡 Future Work", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["💡 Future Work"]
            st.rerun()
        if col4.button("❓ Viva Questions", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["❓ Viva Questions"]
            st.rerun()
        if col5.button("📊 Explain Simply", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["📊 Explain Simply"]
            st.rerun()
        if col6.button("🔍 Research Gaps", use_container_width=True):
            st.session_state["preset_query"] = PRESET_QUERIES["🔍 Research Gaps"]
            st.rerun()
        with center:
            if st.button("🟨 Highlight PDF", use_container_width=True):
                st.session_state["preset_query"] = "__HIGHLIGHT__"
                st.rerun()

        if st.session_state["preset_query"]:
            query = st.session_state["preset_query"]
            st.session_state["preset_query"] = None
            process_query(query, st.session_state["vectorstore"], llm, selected_language, selected_domain)

        query = st.chat_input("Ask a question about your documents (type 'exit' to stop)...")
        if query:
            if query.strip().lower() == "exit":
                st.session_state["conversation_ended"] = True
                st.rerun()
            else:
                process_query(query, st.session_state["vectorstore"], llm, selected_language, selected_domain)