# -*- coding: utf-8 -*-
"""
Drophet — Gradio interface for testing DDI predictions.

Usage:
    python3 drophet_app.py             # open local UI on http://127.0.0.1:7860
    DROPHET_SHARE=1 python3 drophet_app.py   # also expose a 72-hour public tunnel

Wraps the trained GNN inference tool from 019_train_gnn_model.py. The script
filename starts with a digit, so this module loads it via importlib instead of
a regular import.

Requirements:
    - gradio (see requirements.txt)
    - ddi_gnn_best_model.pth in the repo root
        (produced by `python3 019_train_gnn_model.py`)

Not for clinical use. SMILES are resolved live via PubChem REST.
"""

import os
import importlib.util
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
GNN_SCRIPT = REPO_ROOT / "019_train_gnn_model.py"


def _load_gnn_module():
    spec = importlib.util.spec_from_file_location("drophet_gnn", GNN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {GNN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gnn = _load_gnn_module()
_tool = _gnn.DDIInferenceTool()


def predict(drug1: str, drug2: str):
    drug1 = (drug1 or "").strip()
    drug2 = (drug2 or "").strip()

    if not _tool.is_ready:
        msg = (
            "### ⚠️ Model artifacts missing\n\n"
            "Run `python3 019_train_gnn_model.py` first to produce "
            "`ddi_gnn_best_model.pth`."
        )
        return msg, "", "", ""

    if not drug1:
        return "Please enter at least Drug 1.", "", "", ""

    res = _tool.predict_from_names(drug1, drug2)

    if "error" in res:
        return f"### ❌ {res['error']}", "", "", ""

    summary = (
        f"### {res['tier']}\n\n"
        f"**Predicted adverse-event incidence:** {res['incidence']}"
    )
    inputs_view = (
        f"Drug 1: {drug1}\n"
        f"Drug 2: {drug2 if drug2 else '(monotherapy — no second drug)'}"
    )
    return summary, res.get("s1", ""), res.get("s2", ""), inputs_view


with gr.Blocks(title="Drophet — DDI Risk Screening") as demo:
    gr.Markdown(
        "# 💊 Drophet — Drug-Drug Interaction Risk Screening\n"
        "Enter two drug names to get a predicted adverse-event incidence rate "
        "and a coarse risk tier. Leave **Drug 2** blank to score **Drug 1** as "
        "monotherapy. SMILES are fetched from PubChem on submit."
    )

    with gr.Row():
        d1 = gr.Textbox(label="Drug 1", placeholder="e.g. Warfarin", autofocus=True)
        d2 = gr.Textbox(label="Drug 2 (optional)", placeholder="e.g. Aspirin")

    btn = gr.Button("Predict", variant="primary")

    result_md = gr.Markdown()

    with gr.Row():
        s1_box = gr.Textbox(label="SMILES 1 (PubChem)", interactive=False, lines=2)
        s2_box = gr.Textbox(label="SMILES 2 (PubChem)", interactive=False, lines=2)

    info_box = gr.Textbox(label="Inputs", interactive=False, lines=2)

    outputs = [result_md, s1_box, s2_box, info_box]
    btn.click(predict, inputs=[d1, d2], outputs=outputs)
    d1.submit(predict, inputs=[d1, d2], outputs=outputs)
    d2.submit(predict, inputs=[d1, d2], outputs=outputs)

    gr.Examples(
        examples=[
            ["Warfarin", "Aspirin"],
            ["Ibuprofen", "Ketoconazole"],
            ["Metformin", ""],
            ["Simvastatin", "Clarithromycin"],
        ],
        inputs=[d1, d2],
    )

    gr.Markdown(
        "---\n"
        "*Research prototype — not for clinical use. Tier thresholds are "
        "the same as the CLI: <5% Low, <20% Moderate, ≥20% High.*"
    )


if __name__ == "__main__":
    share = os.environ.get("DROPHET_SHARE", "").lower() in ("1", "true", "yes")
    demo.launch(share=share, server_name="127.0.0.1")
