"""Streamlit Web Application for Indian Designs Law Assistant.

Provides an intuitive, beginner-friendly chat interface with:
- Fast ChromaDB document retrieval
- Factually grounded answers exclusively from the 87-page Manual
- Interactive citations with chapter/section references and PDF page numbers
- Quick-click sample buttons for all five assignment questions
- Expandable evidence inspect panels
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import DesignsLawAssistant
from src.vector_store import get_or_create_vector_store

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Designs Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional legal UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 15px;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .not-found-badge {
        color: #B91C1C;
        background-color: #FEF2F2;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ChromaDB vector store and legal index...")
def load_assistant():
    """Cache the assistant instance so it loads once across sessions."""
    vs = get_or_create_vector_store()
    return DesignsLawAssistant(vector_store=vs, k=5)


# Initialize assistant
assistant = load_assistant()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the Indian Designs Law Assistant!**\n\n"
                "I answer questions strictly and exclusively from the official **87-page Manual of Designs Practice & Procedure** "
                "(Office of the Controller General of Patents, Designs & Trade Marks).\n\n"
                "- Every verified answer cites the specific **Chapter, Section, and PDF Page Number**.\n"
                "- If a query is not explicitly supported by the text, I reply strictly: **\"Not found\"**.\n\n"
                "Try one of the quick test cases in the sidebar or ask your own question below!"
            ),
            "citations": [],
            "retrieved_chunks": []
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=64)
    st.title("Document Context")
    
    st.markdown("""
    <div class="badge-card">
        <strong>📄 Manual of Designs Practice & Procedure</strong><br>
        <span style="font-size: 0.85rem; color: #166534;">
            • Total Pages: 87<br>
            • Indexed Chunks: 235<br>
            • Database: ChromaDB (Persistent)<br>
            • Publisher: Office of CGPDTM, India
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Assignment Questions")
    st.caption("Click any question to immediately test the RAG pipeline:")

    test_queries = [
        ("Q1: Copyright Duration & Extension", "How long does a proprietor have copyright in a registered design, and can it be extended?"),
        ("Q2: Classification System", "Which classification system is used to register designs?"),
        ("Q3: Trademark Fee (Out-of-Scope)", "What is the government fee to register a trademark?"),
        ("Q4: Current Controller General", "Who is the current Controller General of Patents?"),
        ("Q5: Design Registration Fee", "What is the fee to register a design?")
    ]

    selected_query = None
    for label, q_text in test_queries:
        if st.button(label, use_container_width=True):
            selected_query = q_text

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Re-index", use_container_width=True):
            with st.spinner("Rebuilding ChromaDB from PDF..."):
                get_or_create_vector_store(force_rebuild=True)
                st.success("Re-indexed!")
                st.rerun()

    st.caption("🔒 Secrets managed via environment variables. Zero hardcoded credentials.")

# ----------------- MAIN CONTENT -----------------
st.markdown('<div class="main-title">⚖️ Indian Designs Law Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Grounded Question Answering over the CGPDTM Manual of Designs Practice & Procedure (87 Pages)</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display citations if present
        if msg.get("citations"):
            with st.expander(f"📚 Verified Citations ({len(msg['citations'])})", expanded=False):
                for cit in msg["citations"]:
                    st.markdown(f"• **{cit['citation']}** *(Similarity distance: `{cit['distance_score']}`)*")

        # Display retrieved chunks if present
        if msg.get("retrieved_chunks") and msg["role"] == "assistant" and msg["content"] != "Not found":
            with st.expander(f"🔎 Inspect Retrieved Evidence ({len(msg['retrieved_chunks'])} chunks)", expanded=False):
                for idx, chunk in enumerate(msg["retrieved_chunks"], 1):
                    meta = chunk.get("metadata", {})
                    st.markdown(f"**Chunk {idx}: {meta.get('citation', 'Reference')}** (Distance: `{chunk.get('score')}`)")
                    st.code(chunk.get("content", ""), language="text")

# Handle user input from chat or sidebar buttons
user_query = st.chat_input("Ask a question about Indian Designs Law...")
active_prompt = selected_query or user_query

if active_prompt:
    # 1. Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": active_prompt
    })
    
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # 2. Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving manual excerpts and verifying evidence..."):
            t_start = time.time()
            result = assistant.answer_question(active_prompt)
            duration = time.time() - t_start

        answer = result["answer"]
        citations = result["citations"]
        chunks = result["retrieved_chunks"]

        # Render answer
        if answer == "Not found":
            st.markdown(
                '<div style="background-color: #FEF2F2; border: 1px solid #FECACA; padding: 12px; border-radius: 8px;">'
                '<strong style="color: #991B1B;">Not found</strong><br>'
                '<span style="font-size: 0.9rem; color: #7F1D1D;">This query is not explicitly supported by the 87-page Manual of Designs. '
                'Following strict hallucination prevention rules, no external information was used.</span>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(answer)

        # Show citations
        if citations:
            with st.expander(f"📚 Verified Citations ({len(citations)})", expanded=True):
                for cit in citations:
                    st.markdown(f"• **{cit['citation']}** *(Distance: `{cit['distance_score']}`)*")

        # Show retrieved chunks
        if chunks:
            with st.expander(f"🔎 Inspect Retrieved Evidence ({len(chunks)} chunks) [Took {duration:.2f}s]", expanded=False):
                for idx, ch in enumerate(chunks, 1):
                    m = ch.get("metadata", {})
                    st.markdown(f"**Chunk {idx}: {m.get('citation', 'Reference')}** (Distance: `{ch.get('score')}`)")
                    st.code(ch.get("content", ""), language="text")

    # 3. Store in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "retrieved_chunks": chunks
    })
