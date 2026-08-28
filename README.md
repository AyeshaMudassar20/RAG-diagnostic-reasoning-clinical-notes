# RAG for Diagnostic Reasoning in Clinical Notes (DiReCT)

A Retrieval-Augmented Generation (RAG) system that answers clinical questions grounded in real discharge-summary notes, built on the MIMIC-IV-Ext DiReCT dataset.

## Overview

Given a clinical question, the system retrieves the most relevant discharge notes from a FAISS vector index built with a biomedical sentence embedding model, then feeds those notes as context to an instruction-tuned language model that is explicitly constrained to answer only from what's in the retrieved context (and to say so plainly when the context doesn't contain an answer). A Streamlit app wraps retrieval and generation into a simple query -> evidence -> answer interface.


## Dataset

[MIMIC-IV-Ext DiReCT](https://physionet.org/) discharge summaries, restructured into 511 preprocessed clinical notes, each split into six labeled sections:

| Section | Source field |
|---|---|
| Chief Complaint / Symptoms | `input1` |
| History of Present Illness | `input2` |
| Past Medical History / Allergies | `input3` |
| Review of Systems | `input4` |
| Physical Exam | `input5` |
| Relevant Labs / Diagnostics | `input6` |

Each note also carries a `diagnosis_category` derived from its folder path in the source dataset (e.g. `Asthma / Severe Asthma Exacerbation`).

**This dataset is not included in this repository, and no real note text appears anywhere in it.** MIMIC-IV-Ext is a credentialed PhysioNet dataset; its data use agreement prohibits public redistribution of note content, including de-identified excerpts. See Data Access & Privacy below.

## Architecture

```
                 User Clinical Query
                          |
                          v
        Dense Retriever (biomedical sentence embeddings)
        - pritamdeka/S-Biomed-Roberta-snli-multinli-stsb
        - 768-dim embeddings, normalized
        - FAISS IndexFlatIP (inner product == cosine on normalized vectors)
                          |
                          v
                 Top-k Retrieved Notes
                          |
                          v
        Generator: microsoft/Phi-3-mini-4k-instruct
        - System prompt restricts answers to facts stated in context
        - Explicitly told to refuse rather than speculate
        - do_sample=True, temperature=0.3, top_p=0.9, max_new_tokens=250
                          |
                          v
                    Final Answer
```

The retrieval and generation prompt (`build_rag_prompt`) is deliberately strict: it forbids the model from mentioning findings not present in the retrieved notes and gives it a fixed refusal string to use when context is insufficient, which is a simple guardrail against hallucinated clinical claims.

## Repository Structure

```
rag-diagnostic-reasoning-clinical-notes/
├── Nlp_Ragg.ipynb        # End-to-end notebook: preprocessing, embeddings, FAISS index,
│                         # retrieval, generation, evaluation (13 steps)
├── app.py                # Streamlit demo: query box, retrieved-evidence viewer, answer
├── data/
│   ├── README.md              # Expected raw dataset layout + preprocessed schema
│   └── sample_documents.jsonl # 3 synthetic example records (schema only, not real notes)
├── requirements.txt
├── LICENSE
└── README.md
```

`index.faiss` and `preprocessed_documents.jsonl` (the compiled vector index and preprocessed notes used at runtime) are not committed -- see below.

## Data Access & Privacy

This project follows MIMIC-IV-Ext's data use agreement:

- No raw or de-identified clinical note text is committed to this repository.
- `preprocessed_documents.jsonl` and `index.faiss` are excluded via `.gitignore` -- both are generated from restricted-access data and are only meant to exist locally, on infrastructure covered by your own PhysioNet credentialing.
- The notebook's saved cell outputs that would have shown real note excerpts (preprocessed note dumps, retrieval previews, generation Q&A pairs, error analysis) have been redacted with a note explaining why; the underlying code is untouched and reproducible.
- `data/sample_documents.jsonl` shows the JSONL schema with three synthetic example records so the format is clear without using real patient data.

To run this project for real, get your own credentialed access to MIMIC-IV-Ext DiReCT via PhysioNet, place the raw dataset as described in `data/README.md`, and run the notebook end-to-end to regenerate `preprocessed_documents.jsonl`, `index.faiss`, and the saved embedding model locally.

## Setup

```bash
git clone https://github.com/AyeshaMudassar20/RAG-diagnostic-reasoning-clinical-notes.git
cd RAG-diagnostic-reasoning-clinical-notes
pip install -r requirements.txt
```

You will also need:

1. Credentialed access to **MIMIC-IV-Ext DiReCT** on PhysioNet (this is a restricted, credentialed dataset -- it is not bundled with this repo).
2. The raw dataset placed under `data/Finished/` as described in `data/README.md`.
3. A GPU is strongly recommended for running `Nlp_Ragg.ipynb` end-to-end (embedding generation and Phi-3-mini inference are both much slower on CPU).

Running the notebook end-to-end produces `preprocessed_documents.jsonl`, `index.faiss`, and a local `embedding_model/` checkpoint -- all required by `app.py` and all excluded from this repo via `.gitignore` (see Data Access & Privacy above).

## Usage

**Notebook (`Nlp_Ragg.ipynb`)** -- run in Google Colab (or locally with a GPU) to reproduce the pipeline end-to-end:

1. Upload your own `Finished.zip` (your credentialed MIMIC-IV-Ext DiReCT export).
2. The notebook extracts it, preprocesses notes into `preprocessed_documents.jsonl`, builds the FAISS index, loads the embedding and generator models, and runs retrieval + generation.
3. A small self-contained evaluation (Step 12) reports retrieval precision/recall and generation quality on a held-out sample.

**Streamlit app (`app.py`)** -- a simple interface over the same retrieval + generation pipeline, once you have `index.faiss` and `preprocessed_documents.jsonl` locally:

```bash
streamlit run app.py
```

Enter a clinical question, and the app will show the retrieved supporting notes alongside a generated, context-grounded answer.

## Evaluation

The notebook includes a lightweight, self-referential evaluation (Step 12), not a rigorous benchmark: 10 documents are sampled, a query is auto-extracted from each (its Chief Complaint, or a fallback sentence), and the retriever is asked to find the source document again among the full 511-note index.

Measured results on that 10-example sample, k=3:

| Metric | Value |
|---|---|
| Retrieval Precision@3 | 0.333 |
| Retrieval Recall@3 | 1.0 |

Recall@3 = 1.0 makes sense given the setup -- the query is derived directly from the same document it should retrieve, so the source document is virtually guaranteed to be a top match. Precision@3 = 0.333 reflects that only 1 of the 3 retrieved slots is the "correct" self-match, with the other 2 being the next-closest notes by embedding similarity (not necessarily wrong, just not the exact source).

This is intentionally simple and should not be read as a claim about real-world retrieval quality against unseen clinical questions -- it mainly verifies that the retrieval pipeline (embedding, indexing, search) is wired correctly end-to-end. Generation quality was checked qualitatively during development rather than with a formal metric; the per-example generation outputs are redacted in the notebook for the same MIMIC-IV-Ext data use agreement reason described above, since they quote retrieved note text.
