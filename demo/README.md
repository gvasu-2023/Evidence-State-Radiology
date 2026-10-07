# Phase 28A local demonstrator

This Streamlit app presents the frozen Phase 26D study records and outputs alongside the existing evidence-state analyzer and reliability gate. It runs locally and does not call MAIRA-2, download models, or perform inference.

## Run on Windows

From the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r demo\requirements.txt
.venv\Scripts\python.exe -m streamlit run demo\app.py
```

Open the local URL printed by Streamlit. The demo reads the Phase 26D generation CSV, the frozen master results table, and the exact image paths recorded in the CSV. If an image or report is unavailable, the app says so rather than substituting another record.

## Suggested walkthrough

1. Choose a quick walkthrough condition or select a study and benchmark condition.
2. Compare the saved reports for lambda 0.00 and the frozen lambda 0.25.
3. Inspect the stored context, then edit it to see the structural completeness parser respond.
4. Set the reviewer-provided relevance, consistency, incompleteness, and evidence-strength inputs. The state and gate panel updates using the existing project code.
5. Review the frozen results snapshot and its recorded interpretation.

The benchmark condition is shown as metadata and is never passed into the state analyzer. The quick buttons initialize transparent reviewer controls for an illustrative walkthrough; those are not automatic classifications.

The analyzer does not infer semantic relevance, conflict, or visual evidence strength from text or pixels. Relevance, consistency, and evidence strength are explicit reviewer inputs. The existing parser detects structural fragmentation in the supplied text; it does not establish clinical completeness. Both Phase 26D incomplete subtypes map to the analyzer's single `INCOMPLETE` state.
