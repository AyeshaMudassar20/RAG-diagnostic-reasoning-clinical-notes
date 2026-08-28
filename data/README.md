# Data Directory

This directory intentionally does not contain the MIMIC-IV-Ext DiReCT dataset or anything derived from real patient notes. See the top-level README's "Data Access & Privacy" section for why.

## Expected layout for the raw dataset

If you have your own credentialed PhysioNet access to MIMIC-IV-Ext DiReCT, place the raw dataset here before running `Nlp_Ragg.ipynb`:

```
data/
└── Finished/
    └── <Diagnosis Category>/
        └── <Sub-category, optional>/
            └── <note-id>.json   # fields: input1..input6 (see below)
```

Each raw JSON file is expected to have up to six free-text fields, which the notebook's `parse_json_file()` maps to labeled note sections:

| Field | Mapped section |
|---|---|
| `input1` | Chief Complaint / Symptoms |
| `input2` | History of Present Illness |
| `input3` | Past Medical History / Allergies |
| `input4` | Review of Systems |
| `input5` | Physical Exam |
| `input6` | Relevant Labs / Diagnostics |

The notebook walks `data/Finished/` recursively, builds a `diagnosis_category` from each file's parent folder path, and writes the result to `preprocessed_documents.jsonl` (not committed -- see schema example below).

## Preprocessed schema (synthetic example)

`sample_documents.jsonl` in this directory shows the exact JSONL schema produced by the notebook, using three made-up records -- no real patient data:

```json
{"id": "example-0001", "diagnosis_category": "Example Category", "text": "[Chief Complaint / Symptoms]:\n...", "file_path": "data/Finished/Example Category/example-0001.json"}
```

Run the notebook against your own licensed copy of the dataset to produce the real `preprocessed_documents.jsonl` and `index.faiss`.

