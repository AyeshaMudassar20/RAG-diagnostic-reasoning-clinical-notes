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
