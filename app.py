import streamlit as st

from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain,
)






def main():
    st.set_page_config(
        page_title="Information Retrieval System",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Information Retrieval System 🤖")
    st.write("Upload your PDF and ask questions about it.")

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # Sidebar
    with st.sidebar:
        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)

                    text_chunks = get_text_chunks(raw_text)

                    vector_store = get_vector_store(text_chunks)

                    llm, retriever = get_conversational_chain(
                        vector_store
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.llm = llm

                st.success("PDF processing completed!")

    # Question input
    user_question = st.text_input(
        "Ask a question from your PDF"
    )

    if user_question:
        if st.session_state.retriever is None:
            st.warning("Please upload and process a PDF first.")
        else:
            with st.spinner("Generating answer..."):
                documents = st.session_state.retriever.invoke(
                    user_question
                )

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                prompt = f"""
You are a helpful assistant answering questions
based only on the provided PDF context.

If the answer is not present in the context,
say that you could not find the answer in the
uploaded documents.

Context:
{context}

Question:
{user_question}

Answer:
"""

                response = st.session_state.llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response.content[0]["text"])


if __name__ == "__main__":
    main()