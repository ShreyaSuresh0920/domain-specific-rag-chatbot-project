# Domain-Specific RAG Chatbot — Project Report

## Objective

The project implements a PDF question-answering assistant that helps users locate information in large documents. The application preserves page-level source metadata, retrieves semantically related passages, and presents an answer in a simple two-sided chat interface.

## Implemented workflow

The application extracts readable PDF pages with `pypdf`, records each filename and page number, splits text into overlapping chunks, generates embeddings with `all-MiniLM-L6-v2`, and stores those embeddings in FAISS. The index is saved locally and restored automatically when the application restarts.

For each question, the application retrieves the four most relevant chunks. Groq is used as an optional strict grounded-answer layer. Gemini receives the retrieved answer, raw PDF passages, source metadata, and recent conversation history. It must then produce an **Answer from the knowledge base**, an **In simple words** explanation, and **Extra context** for general questions or useful background.

## User interface

The Streamlit interface includes multi-conversation chat history, a new-chat control, clear-current-chat control, PDF upload validation, indexing feedback, connection status, a restored-index status pill, source expanders, and a polished message layout with user messages on the right and assistant messages on the left.

## Safety and reliability controls

API keys remain in `.env` and are never embedded into Python source. The `.gitignore` file excludes local credentials, Streamlit secrets, generated Python files, and the saved FAISS index. PDF uploads are limited to PDF files, ten files per indexing action, and 25 MB per file. Retrieved document text is treated as untrusted content, and prompts explicitly ignore instructions inside documents that attempt to alter assistant behavior.

When the PDF does not contain the requested fact, the first response section says so rather than inventing a document answer. General knowledge is allowed only in the separately labeled extra-context section. High-stakes information should still be verified against original documents and qualified sources.

## Validation completed

The supplied PDF produced eight readable pages and fifteen chunks. The project passed Python syntax validation, PDF extraction and prompt smoke tests, FAISS retrieval validation, and a live application-flow test. The rendered Streamlit app was inspected with a restored fifteen-chunk index. A document-grounded question, a general embedding question, and an unavailable CEO question all produced the required structured response and source display. One Streamlit avatar compatibility issue found during visual testing was fixed by using the built-in assistant avatar.

The repository includes `tests/test_questions.csv` with fifteen evaluation questions covering correct answers, unavailable information, general questions, safety behavior, and source verification.
