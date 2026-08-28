# RAG for Diagnostic Reasoning in Clinical Notes (DiReCT)

A Retrieval-Augmented Generation (RAG) system that answers clinical questions grounded in real discharge-summary notes, built on the MIMIC-IV-Ext DiReCT dataset.

## Overview

Given a clinical question, the system retrieves the most relevant discharge notes from a FAISS vector index built with a biomedical sentence embedding model, then feeds those notes as context to an instruction-tuned language model that is explicitly constrained to answer only from what's in the retrieved context (and to say so plainly when the context doesn't contain an answer). A Streamlit app wraps retrieval and generation into a simple query -> evidence -> answer interface.

