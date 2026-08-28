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
