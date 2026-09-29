import os
import tempfile
import time

import streamlit as st

from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import create_embedding
from src.vector_store import VectorStore
from src.rag import generate_answer, generate_comparison


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/index/metadata.pkl"


def generate_with_retry(generator, *args, max_retries=2, **kwargs):
    """Retry transient Gemini service errors before showing an error."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return generator(*args, **kwargs)
        except Exception as error:
            last_error = error
            message = str(error).lower()

            transient = (
                "503" in message
                or "service unavailable" in message
                or "temporarily unavailable" in message
                or "429" in message
                or "resource exhausted" in message
            )

            if not transient or attempt == max_retries:
                raise

            time.sleep(2 ** attempt)

    raise last_error


def clean_generated_comparison(text):
    """Clean common HTML artifacts produced inside Markdown tables."""
    return (
        text
        .replace("<br>", ", ")
        .replace("<br/>", ", ")
        .replace("<br />", ", ")
    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .status-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #123c2a;
        border: 1px solid #1f7a52;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions across multiple PDFs using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Documents")

    st.subheader("🔎 Mode")

    mode = st.radio(
        "Choose how you want to use the assistant:",
        [
            "💬 Ask Questions",
            "📊 Compare Documents"
        ],
        label_visibility="collapsed"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "⚙️ Process PDFs",
        use_container_width=True
    )

    st.divider()


    # --------------------------------------------------------
    # Display selected files
    # --------------------------------------------------------

    if uploaded_files:

        st.markdown("### Selected PDFs")

        for file in uploaded_files:

            st.markdown(
                f"📄 `{file.name}`"
            )


    # --------------------------------------------------------
    # Current knowledge base
    # --------------------------------------------------------

    if st.session_state.pdf_processed:

        st.divider()

        st.success(
            "Knowledge base ready"
        )

        st.markdown(
            "### Current documents"
        )

        for name in st.session_state.document_names:

            st.markdown(
                f"📄 `{name}`"
            )


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "PDFs",
                len(
                    st.session_state.document_names
                )
            )

        with col2:

            st.metric(
                "Chunks",
                st.session_state.chunk_count
            )


        st.metric(
            "Total pages",
            st.session_state.page_count
        )


        st.divider()


        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


    else:

        st.info(
            "Upload one or more PDFs and click "
            "**Process PDFs** to create your knowledge base."
        )


# ============================================================
# PROCESS MULTIPLE PDFs
# ============================================================

if uploaded_files and process_button:

    temp_files = []

    try:

        with st.spinner(
            "Processing your PDF collection..."
        ):

            all_chunks = []

            total_pages = 0

            document_names = []


            # ==================================================
            # PROCESS EACH PDF
            # ==================================================

            for file_number, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):

                st.write(
                    f"📄 Processing "
                    f"{file_number}/{len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                )


                # ------------------------------------------------
                # Save temporary PDF
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = (
                        temp_file.name
                    )

                temp_files.append(
                    temp_pdf_path
                )


                # ------------------------------------------------
                # Extract text
                # ------------------------------------------------

                pages = extract_text_from_pdf(
                    temp_pdf_path
                )

                if not pages:

                    raise ValueError(
                        f"No readable text found in "
                        f"{uploaded_file.name}"
                    )


                total_pages += len(
                    pages
                )

                document_names.append(
                    uploaded_file.name
                )


                # ------------------------------------------------
                # Create chunks
                # ------------------------------------------------

                chunks = chunk_pages(
                    pages
                )


                # ------------------------------------------------
                # Add source information
                # ------------------------------------------------

                for chunk in chunks:

                    chunk["source"] = (
                        uploaded_file.name
                    )

                all_chunks.extend(
                    chunks
                )


            # ==================================================
            # CHECK CHUNKS
            # ==================================================

            if not all_chunks:

                raise ValueError(
                    "No usable text chunks were created."
                )


            st.write(
                f"📄 Total pages: {total_pages}"
            )

            st.write(
                f"✂️ Total chunks: "
                f"{len(all_chunks)}"
            )


            # ==================================================
            # GENERATE EMBEDDINGS
            # ==================================================

            progress = st.progress(
                0,
                text="Creating embeddings..."
            )

            embeddings = []


            for i, chunk in enumerate(
                all_chunks
            ):

                embedding = create_embedding(
                    chunk["text"]
                )

                embeddings.append(
                    embedding
                )

                progress.progress(
                    (i + 1) / len(all_chunks),
                    text=(
                        f"Embedding chunk "
                        f"{i + 1}/"
                        f"{len(all_chunks)}"
                    )
                )


            # ==================================================
            # CREATE FAISS INDEX
            # ==================================================

            dimension = len(
                embeddings[0]
            )

            vector_store = VectorStore(
                dimension
            )


            vector_store.add_embeddings(
                embeddings,
                all_chunks
            )


            # ==================================================
            # SAVE INDEX
            # ==================================================

            vector_store.save(
                INDEX_PATH,
                METADATA_PATH
            )


            # ==================================================
            # UPDATE SESSION STATE
            # ==================================================

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.pdf_processed = (
                True
            )

            st.session_state.document_names = (
                document_names
            )

            st.session_state.page_count = (
                total_pages
            )

            st.session_state.chunk_count = (
                len(all_chunks)
            )

            # New knowledge base = new conversation
            st.session_state.messages = []


        st.success(
            f"✅ Successfully processed "
            f"{len(document_names)} PDF(s)!"
        )

        st.rerun()


    except Exception as e:

        st.error(
            f"❌ Error processing PDFs: {e}"
        )


    finally:

        for temp_file in temp_files:

            if os.path.exists(temp_file):

                os.remove(
                    temp_file
                )


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

if (
    st.session_state.vector_store is None
    and os.path.exists(INDEX_PATH)
    and os.path.exists(METADATA_PATH)
):

    try:

        vector_store = VectorStore.load(
            INDEX_PATH,
            METADATA_PATH
        )


        # ----------------------------------------------------
        # Recover document names from metadata
        # ----------------------------------------------------

        document_names = sorted(
            {
                metadata.get(
                    "source",
                    "Unknown document"
                )
                for metadata
                in vector_store.metadata
            }
        )


        # ----------------------------------------------------
        # Recover page count
        # ----------------------------------------------------

        pages = set()

        for metadata in vector_store.metadata:

            source = metadata.get(
                "source",
                "Unknown document"
            )

            page = metadata.get(
                "page_number"
            )

            pages.add(
                (
                    source,
                    page
                )
            )


        # ----------------------------------------------------
        # Update session state
        # ----------------------------------------------------

        st.session_state.vector_store = (
            vector_store
        )

        st.session_state.pdf_processed = (
            True
        )

        st.session_state.document_names = (
            document_names
        )

        st.session_state.page_count = (
            len(pages)
        )

        st.session_state.chunk_count = (
            vector_store.index.ntotal
        )


    except Exception:

        st.session_state.vector_store = None

        st.session_state.pdf_processed = (
            False
        )


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.pdf_processed:

    st.markdown(
        """
        <div class="status-card">
        🟢 <b>Knowledge base ready</b> —
        You can ask questions across your uploaded PDFs.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if (
            role == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📚 View sources"
            ):

                displayed_sources = set()


                for source in message["sources"]:

                    document = source.get(
                        "source",
                        "Unknown document"
                    )

                    page = source.get(
                        "page_number",
                        "?"
                    )

                    source_key = (
                        document,
                        page
                    )


                    if source_key in displayed_sources:

                        continue


                    displayed_sources.add(
                        source_key
                    )


                    st.markdown(
                        f"📄 **{document}** "
                        f"— Page {page}"
                    )


# ============================================================
# EMPTY CHAT STATE
# ============================================================

if (
    st.session_state.pdf_processed
    and not st.session_state.messages
):

    st.info(
        "💬 Ask a question below to start "
        "chatting with your document collection."
    )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.pdf_processed:

    # ========================================================
    # ASK QUESTIONS MODE
    # ========================================================

    if mode == "💬 Ask Questions":

        question = st.chat_input(
            "Ask something about your PDFs..."
        )

        if question:

            # ====================================================
            # SAVE USER MESSAGE
            # ====================================================

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            # ====================================================
            # DISPLAY USER MESSAGE
            # ====================================================

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # ====================================================
            # ASSISTANT RESPONSE
            # ====================================================

            with st.chat_message("assistant"):

                try:

                    with st.spinner(
                        "Searching your documents..."
                    ):

                        # =========================================
                        # RETRIEVAL QUERY
                        # =========================================

                        retrieval_query = question


                        # =========================================
                        # QUERY EMBEDDING
                        # =========================================

                        query_embedding = (
                            create_embedding(
                                retrieval_query
                            )
                        )


                        # =========================================
                        # FAISS RETRIEVAL
                        # =========================================

                        retrieved_chunks = (
                            st.session_state.vector_store.search(
                                query_embedding,
                                top_k=5
                            )
                        )


                        # =========================================
                        # GENERATE ANSWER
                        # =========================================

                        answer = generate_with_retry(
                            generate_answer,
                            question,
                            retrieved_chunks,
                            chat_history=(
                                st.session_state.messages
                            )
                        )


                    # =============================================
                    # DISPLAY ANSWER
                    # =============================================

                    st.markdown(
                        answer
                    )


                    # =============================================
                    # DISPLAY SOURCES
                    # =============================================

                    with st.expander(
                        "📚 View sources"
                    ):

                        displayed_sources = set()


                        for source in retrieved_chunks:

                            document = source.get(
                                "source",
                                "Unknown document"
                            )

                            page = source.get(
                                "page_number",
                                "?"
                            )


                            source_key = (
                                document,
                                page
                            )


                            if source_key in displayed_sources:

                                continue


                            displayed_sources.add(
                                source_key
                            )


                            st.markdown(
                                f"📄 **{document}** "
                                f"— Page {page}"
                            )


                    # =============================================
                    # SAVE ASSISTANT MESSAGE
                    # =============================================

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks
                        }
                    )


                except Exception as e:

                    error_message = str(e)


                    # ---------------------------------------------
                    # Friendly API errors
                    # ---------------------------------------------

                    if "503" in error_message:

                        friendly_message = (
                            "⚠️ The AI service is temporarily busy. "
                            "I retried the request automatically, but the "
                            "service is still unavailable. Please try again "
                            "in a few seconds."
                        )


                    elif "429" in error_message:

                        friendly_message = (
                            "⚠️ The AI service rate limit was reached. "
                            "I retried the request automatically, but the "
                            "limit is still active. Please wait a moment "
                            "and try again."
                        )


                    else:

                        friendly_message = (
                            f"⚠️ Something went wrong while "
                            f"generating the answer: {error_message}"
                        )


                    st.error(
                        friendly_message
                    )


    # ========================================================
    # DOCUMENT COMPARISON MODE
    # ========================================================

    else:

        st.subheader("📊 Compare Documents")

        if len(st.session_state.document_names) < 2:

            st.warning(
                "Please process at least two PDFs to use "
                "Document Comparison mode."
            )

        else:

            st.info(
                "Ask a question that requires information "
                "from multiple uploaded PDFs."
            )

            comparison_question = st.chat_input(
                "Example: Compare RNN and LSTM according to the uploaded documents..."
            )

            if comparison_question:

                # ====================================================
                # SAVE USER MESSAGE
                # ====================================================

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": comparison_question
                    }
                )


                # ====================================================
                # DISPLAY USER MESSAGE
                # ====================================================

                with st.chat_message("user"):

                    st.markdown(
                        comparison_question
                    )


                # ====================================================
                # COMPARISON RESPONSE
                # ====================================================

                with st.chat_message("assistant"):

                    try:

                        with st.spinner(
                            "Retrieving information from each document..."
                        ):

                            # =========================================
                            # QUERY EMBEDDING
                            # =========================================

                            query_embedding = (
                                create_embedding(
                                    comparison_question
                                )
                            )


                            # =========================================
                            # RETRIEVE MORE CHUNKS
                            #
                            # A larger top_k gives the comparison
                            # a better chance of finding evidence
                            # from multiple PDFs.
                            # =========================================

                            retrieved_chunks = (
                                st.session_state.vector_store.search(
                                    query_embedding,
                                    top_k=20
                                )
                            )


                            # =========================================
                            # GROUP RETRIEVED CHUNKS BY DOCUMENT
                            # =========================================

                            grouped_chunks = {}

                            for chunk in retrieved_chunks:

                                document = chunk.get(
                                    "source",
                                    "Unknown document"
                                )

                                if document not in grouped_chunks:

                                    grouped_chunks[document] = []


                                # Keep the strongest few chunks
                                # from each document.
                                if len(
                                    grouped_chunks[document]
                                ) < 4:

                                    grouped_chunks[
                                        document
                                    ].append(chunk)


                            # =========================================
                            # MAKE SURE MULTIPLE DOCUMENTS
                            # ARE REPRESENTED
                            # =========================================

                            if len(grouped_chunks) < 2:

                                st.warning(
                                    "The retrieved information came "
                                    "from only one document. Try asking "
                                    "a question that explicitly involves "
                                    "both uploaded PDFs."
                                )

                            else:

                                # =====================================
                                # GENERATE COMPARISON
                                # =====================================

                                comparison = generate_with_retry(
                                    generate_comparison,
                                    comparison_question,
                                    grouped_chunks
                                )

                                comparison = clean_generated_comparison(
                                    comparison
                                )


                                # =====================================
                                # DISPLAY COMPARISON
                                # =====================================

                                st.markdown(
                                    comparison
                                )


                                # =====================================
                                # DISPLAY SOURCES
                                # =====================================

                                with st.expander(
                                    "📚 View comparison sources"
                                ):

                                    for (
                                        document,
                                        chunks
                                    ) in grouped_chunks.items():

                                        st.markdown(
                                            f"### 📄 {document}"
                                        )

                                        displayed_pages = set()


                                        for chunk in chunks:

                                            page = chunk.get(
                                                "page_number",
                                                "?"
                                            )


                                            if page in displayed_pages:
                                                continue


                                            displayed_pages.add(
                                                page
                                            )


                                            st.markdown(
                                                f"- Page {page}"
                                            )


                                # =====================================
                                # SAVE COMPARISON IN CHAT HISTORY
                                # =====================================

                                comparison_sources = []

                                for chunks in grouped_chunks.values():

                                    comparison_sources.extend(
                                        chunks
                                    )


                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": comparison,
                                        "sources": comparison_sources
                                    }
                                )


                    except Exception as e:

                        error_message = str(e)


                        # ---------------------------------------------
                        # Friendly API errors
                        # ---------------------------------------------

                        if "503" in error_message:

                            friendly_message = (
                                "⚠️ The AI service is temporarily "
                                "busy. Please try again in a few seconds."
                            )


                        elif "429" in error_message:

                            friendly_message = (
                                "⚠️ The AI service rate limit was "
                                "reached. Please wait a moment and "
                                "try again."
                            )


                        else:

                            friendly_message = (
                                f"⚠️ Something went wrong while "
                                f"generating the comparison: {error_message}"
                            )


                        st.error(
                            friendly_message
                        )
