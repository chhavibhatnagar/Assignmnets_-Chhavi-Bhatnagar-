from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

DATA_DIR = "data"


def load_documents(data_dir=DATA_DIR):
    """Load all PDFs from the data folder."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller overlapping chunks.

    Larger chunks (vs. the original 500/50) reduce mid-sentence
    fragmentation, which was causing the LLM to either hedge
    ("not explicitly stated") or drift outside the context to fill gaps —
    both of which hurt RAGAS faithfulness/relevancy scores.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(chunks, embedding_model):
    """Embed chunks and store them in a FAISS vector store."""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


def build_vectorstore_chroma(chunks, embedding_model, embedding_name="default"):
    """Embed chunks and store them in a Chroma vector store."""
    safe_name = embedding_name.replace("/", "_")
    persist_directory = f"chroma_db_{safe_name}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore


def save_vectorstore(vectorstore, path="faiss_index"):
    """Save FAISS index to disk so we don't have to re-embed every time."""
    vectorstore.save_local(path)


def load_vectorstore(embedding_model, path="faiss_index"):
    """Load a previously saved FAISS index."""
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)


def retrieve(query, vectorstore, k=8, per_doc_k=2):
    """
    Retrieve relevant chunks for a query, guaranteeing representation from
    every uploaded document rather than letting one large document dominate.
    """
    pool_size = max(k * 4, 30)
    pooled = vectorstore.similarity_search_with_score(query, k=pool_size)

    by_source = {}
    for doc, score in pooled:
        source = doc.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append((doc, score))

    num_sources = max(len(by_source), 1)
    effective_per_doc_k = max(per_doc_k, -(-k // num_sources))  

    final = []
    seen = set()
    for source, doc_scores in by_source.items():
        doc_scores.sort(key=lambda x: x[1])
        for doc, score in doc_scores[:effective_per_doc_k]:
            normalized = doc.page_content.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                doc.metadata["similarity_score"] = score
                final.append(doc)

    final.sort(key=lambda doc: doc.metadata.get("similarity_score", float("inf")))
    return final


def get_llm():
    """Load the Groq LLM."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def extract_headings(page_texts):
    """Use OpenAI to extract important section headings from PDF page text."""

    heading_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are given the raw text of a PDF, page by page.

Return ONLY real structural section headings and subsection headings — the
kind that would appear in a table of contents.

STRICT RULES:
- A heading is SHORT: a title or label, never a full sentence.
- NEVER return a sentence, a claim, or anything containing a period in the
  middle of it (e.g. "Results showed significant reductions" is NOT a
  heading — reject it).
- Copy the heading text EXACTLY as it appears in the page text below —
  same words, same capitalization, same punctuation. Do not paraphrase.
- Ignore the document title, author names, affiliations, emails, page
  numbers, running headers/footers, references, and figure/table captions.
- Ignore numbered citation markers like [1] or [12,13].
- If a heading has a number/letter prefix (e.g. "III.", "A."), you may
  include or omit the prefix, but the rest must match the source text
  exactly either way.
- When in doubt whether something is a heading or body text, leave it out.

Return JSON only, no other text.

Example of GOOD output:
[
  {{"heading": "Introduction", "page": 1}},
  {{"heading": "Community Cloud", "page": 5}}
]

Example of BAD output (do NOT do this — these are sentences, not headings):
[
  {{"heading": "Results showed significant reductions in depression", "page": 3}}
]

PDF:

{json.dumps(page_texts)}
"""

    response = heading_llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    if not content:
        return []

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []


def get_document_headings(retrieved_docs, llm):
    """Extract important headings from the retrieved documents using Groq."""
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = ChatPromptTemplate.from_template(
        """You are reading a document.

Extract ONLY the important headings, section titles, or chapter names.

Rules:
- Return one heading per line.
- Do NOT explain anything.
- Do NOT number them.
- Do NOT return paragraphs.
- If no headings exist, return the most important topic names.

Context:
{context}"""
    )

    chain = prompt | llm
    response = chain.invoke({"context": context})

    highlights = []
    for line in response.content.split("\n"):
        line = (
            line.replace("•", "")
                .replace("-", "")
                .replace("1.", "").replace("2.", "").replace("3.", "")
                .strip()
        )
        if line:
            highlights.append(line)

    highlights = list(dict.fromkeys(highlights))
    return highlights


def generate_answer(query, retrieved_docs, llm, chat_history=None, response_language="Auto-detect", domain="General"):
    """Generate an answer grounded in retrieved context, with language and domain awareness."""
    context_parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")
        context_parts.append(f"[SOURCE: {source}]\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    history_text = ""
    if chat_history:
        history_lines = []
        for turn in chat_history[-3:]:
            history_lines.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        history_text = "\n\n".join(history_lines)

    if response_language == "Auto-detect":
        language_instruction = "Respond in the SAME language the question is written in."
    else:
        language_instruction = f"Respond in {response_language}, regardless of what language the question is written in."

    domain_instructions = {
        "General": "",
        "Legal": "You are assisting with legal document analysis. Use precise legal language, reference specific clauses or sections when relevant, and always include a disclaimer that this is not formal legal advice.",
        "Medical": "You are assisting with medical document analysis. Use proper clinical terminology, be accurate and conservative in your statements, and always recommend consulting a qualified healthcare professional for medical decisions.",
        "Finance": "You are assisting with financial document analysis. Be precise with numbers and figures, clearly state any assumptions made, and note relevant financial risks or limitations where applicable.",
        "Research": "You are assisting with academic research analysis. Use formal academic tone, reference methodology and findings precisely, and flag any limitations, gaps, or areas needing further study.",
    }
    domain_instruction = domain_instructions.get(domain, "")

    prompt = ChatPromptTemplate.from_template(
        """You are a document question answering assistant.

Answer ONLY using the retrieved context provided below.

IMPORTANT RULES:

1. Use ONLY information explicitly present in the retrieved context.

2. Never use outside knowledge, assumptions, or guesses.

3. If the retrieved context does not explicitly contain the answer to the user's question, respond EXACTLY with:

"I couldn't find information about this in the uploaded document(s). Please ask a question related to the uploaded document."

4. Do NOT provide related information.

5. Do NOT summarize other parts of the document.

6. Do NOT infer or complete missing information.

7. If the user's question is unrelated to the uploaded document(s), use the exact response above.

""" + language_instruction + """

""" + (domain_instruction + "\n\n" if domain_instruction else "") + """Each piece of context is tagged with [SOURCE: filename] showing which file it came from. Use these tags to know which pieces belong together.

IMPORTANT: Organize your answer with exactly ONE header per unique [SOURCE: ...] filename — never more than one, no matter how many times that filename appears in the context or how repetitive its content is. NEVER create headers like "(Duplicate Section)", "(Section 3)", or any variation suggesting multiple parts of the same file. If you see the same filename tagged multiple times, silently merge all of it into the ONE header for that file, removing any repeated information. Combine ALL content tagged with the same filename under that file's single header, in your own words. Use a clean readable name based on the filename (e.g., "HR_Policy_Handbook.pdf" becomes "HR Policy Handbook"), never the raw filename with underscores or extension.

For headers, use **bold text** only (like **Header Name**), never markdown heading syntax like # or ##.

Do not mention the [SOURCE: ...] tags themselves in your answer — they are for your reference only.

If asked for a specific number of points per source, give exactly that many points under each source's single header.

Give a precise, direct answer using ONLY what is explicitly stated in the context. Do not infer, speculate, or add information beyond what the context directly says. If something is not explicitly stated, say so — do not fill gaps with assumptions.

Recent conversation (use this ONLY to understand follow-up questions like "it" or "that" — do not repeat or restate this history in your answer):
{history}

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query, "history": history_text})
    return response.content


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} pages from PDFs")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Building vector store...")
    vectorstore = build_vectorstore(chunks, embedding_model)
    save_vectorstore(vectorstore)
    print("Vector store saved to faiss_index/")

    query = "Why is ai used?"
    print(f"\nQuery: {query}")
    results = retrieve(query, vectorstore)

    print("Loading LLM...")
    llm = get_llm()

    print("Generating answer...")
    answer = generate_answer(query, results, llm)
    print(f"\nAnswer: {answer}")