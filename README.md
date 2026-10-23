# Designs Law Assistant

A Generative AI application that answers questions using the **Manual of Designs Practice and Procedure** PDF. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document sections and generate grounded answers with source references.

## Features

- PDF document processing and text chunking
- Vector search using ChromaDB
- Gemini-powered question answering
- Answers grounded in the uploaded manual
- Source references for relevant answers
- Returns `Not found` when the answer is not supported by the retrieved content

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- LangChain
- ChromaDB
- Retrieval-Augmented Generation (RAG)

## Project Workflow

1. Load the Designs Manual PDF.
2. Extract text and divide it into chunks.
3. Generate embeddings and store them in ChromaDB.
4. Retrieve relevant chunks for a user's question.
5. Send the retrieved context to Gemini.
6. Display the answer with source references.
7. Return `Not found` when the required information cannot be found in the available context.

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/knkabir327-sketch/GenAI_MidTerm.git
cd GenAI_MidTerm
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your own Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Security:** Never upload your real API key or `.env` file to GitHub.

## Run the Application

```bash
streamlit run app.py
```

## Run Tests

```bash
python test_pipeline.py
```

## Sample Questions and Results

1. **What is the term of copyright protection for a registered design?**  
   The application retrieved information from the manual stating a term of 10 years, extendable by 5 years, with references to the relevant sections.

2. **How are designs classified?**  
   The application identified the Locarno Classification and the Third Schedule.

3. **What is the trademark registration fee?**  
   `Not found` — the question is outside the scope of the Designs Manual content retrieved.

4. **Who is the current Controller General?**  
   `Not found` — the retrieved manual content did not establish the current officeholder.

5. **What is the design registration fee?**  
   The application identified the First Schedule of the Designs Rules, 2001, as the source for the prescribed fee. An exact numerical amount was not retrieved.

## Testing

The automated pipeline test suite passed **5 out of 5 tests** during local testing.

## Disclaimer

Answers depend on the content retrieved from the supplied manual. Verify important legal information against official sources.

## Author

Abdul Kabir Khan
