#!/usr/bin/env python3
"""Build the single-file Kaggle notebook from the Jupytext sources and lab22 package.

Tailored for Kaggle environment:
- Sets working directory to /kaggle/working/lab22 so artifacts are saved to Kaggle Output
- Adds an automated bundle exporter at the end (lab22_submission_bundle.zip)
- Does not modify any existing Colab files or notebooks
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKDIR = "/kaggle/working/lab22"
OUT_DIR = REPO / "kaggle"
OUT_FILE = OUT_DIR / "Lab22_DPO_Kaggle.ipynb"

STAGES = [
    ("00_dpo_loss_from_scratch", "core · CPU"),
    ("01_sft_mini", "core"),
    ("02_preference_data", "core"),
    ("03_dpo_train", "core"),
    ("03b_dpo_variants", "bonus"),
    ("04_compare_and_eval", "core"),
    ("05_merge_deploy_gguf", "bonus"),
    ("06_benchmark", "bonus"),
    ("07_grpo_bonus", "bonus"),
]
CELL = re.compile(r"^# %%(?P<md> \[markdown\])?.*$", re.MULTILINE)


def requirements() -> list[str]:
    """Requirement specs from requirements.txt, minus test/notebook tooling."""
    skip = ("jupyterlab", "jupytext", "pytest")
    specs = []
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and not spec.startswith(skip):
            specs.append(spec)
    return specs


def source_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


def percent_cells(path: Path) -> list[dict]:
    """Parse a Jupytext py:percent file (header block dropped)."""
    text = path.read_text(encoding="utf-8")
    marks = list(CELL.finditer(text))
    cells = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() :end].strip("\n")
        if not body.strip():
            continue
        if m.group("md"):
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(md(body))
        else:
            cells.append(code(body))
    return cells


RELEASE_GPU = (
    "# Free GPU memory held by the previous stage (model, trainer, llama.cpp handle).\n"
    "import gc\n"
    "for _name in ('trainer', 'model', 'ref_model', 'llm', 'policy', 'ref', 'tokenizer'):\n"
    "    globals().pop(_name, None)\n"
    "gc.collect()\n"
    "try:\n"
    "    import torch\n"
    "    if torch.cuda.is_available():\n"
    "        torch.cuda.empty_cache()\n"
    "        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')\n"
    "except ImportError:\n"
    "    pass"
)

ZIP_BUNDLE_CODE = (
    "# %% Tự động nén toàn bộ kết quả nộp bài ra /kaggle/working/ để tải về từ tab Output\n"
    "import zipfile\n"
    "from pathlib import Path\n"
    "\n"
    "work = Path('/kaggle/working/lab22')\n"
    "bundle_path = Path('/kaggle/working/lab22_submission_bundle.zip')\n"
    "\n"
    "print('Đang đóng gói artifacts nộp bài...')\n"
    "with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
    "    for folder_name in ['submission', 'data/eval']:\n"
    "        folder = work / folder_name\n"
    "        if folder.exists():\n"
    "            for p in folder.rglob('*'):\n"
    "                if p.is_file():\n"
    "                    zf.write(p, arcname=str(p.relative_to(work)))\n"
    "    adapters = work / 'adapters'\n"
    "    if adapters.exists():\n"
    "        for p in adapters.rglob('*.json'):\n"
    "            zf.write(p, arcname=str(p.relative_to(work)))\n"
    "\n"
    "if bundle_path.exists():\n"
    "    print(f'✓ Đã tạo thành công: {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)')\n"
    "    print('BẠN CHỈ CẦN TẢI FILE lab22_submission_bundle.zip TỪ TAB OUTPUT CỦA KAGGLE VỀ MÁY!')\n"
    "else:\n"
    "    print('Không tìm thấy file kết quả để nén.')\n"
)


def render_kaggle() -> dict:
    pins = " ".join(f'"{s}"' for s in requirements())
    cells = [
        md(
            "# Lab 22 — DPO/ORPO Alignment (Kaggle T4 Edition)\n\n"
            "**Track 3 · Day 22 · VinUni AICB.** Tối ưu hoá riêng cho môi trường Kaggle.\n\n"
            "> **LƯU Ý QUAN TRỌNG TRƯỚC KHI CHẠY TRÊN KAGGLE:**\n"
            "> 1. **BẬT INTERNET:** Nhìn sang thanh panel **Settings** (bên phải) -> bật **Internet: ON**.\n"
            "> 2. **CHỌN GPU:** Chọn **Accelerator: GPU T4 x 2** (hoặc GPU T4 x 1).\n"
            "> 3. **CHẠY TẮT MÁY (BACKGROUND):** Bấm **Save Version** (góc trên phải) -> chọn **Save & Run All (Commit)**.\n"
            ">    Khi chạy chế độ này, bạn có thể tắt máy, đóng trình duyệt và đi ngủ! Khi xong, file kết quả sẽ nằm ở tab **Output**.\n"
        ),
        md("## A. Setup cho Kaggle"),
        code(
            "import os\n"
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
            'os.environ["COMPUTE_TIER"] = "T4"\n'
            "# Đảm bảo chạy được ngay cả khi không có khoá API bí mật\n"
        ),
        code(f"!pip install -q {pins}"),
        code(
            "from pathlib import Path\n"
            f'WORK = Path("{WORKDIR}")\n'
            '(WORK / "lab22").mkdir(parents=True, exist_ok=True)\n'
            "os.chdir(WORK)\n"
            "print(f'Thư mục làm việc: {Path.cwd()}')"
        ),
        md("### Helper package `lab22/` (tự động ghi vào `/kaggle/working/lab22`)"),
    ]
    for module in sorted((REPO / "lab22").glob("*.py")):
        body = module.read_text(encoding="utf-8")
        cells.append(code(f"%%writefile {WORKDIR}/lab22/{module.name}\n{body}"))
    for i, (stem, kind) in enumerate(STAGES):
        if i:
            cells.append(code(RELEASE_GPU))
        cells.append(md(f"---\n# ⏵ `notebooks/{stem}.py` ({kind})"))
        cells.extend(percent_cells(REPO / "notebooks" / f"{stem}.py"))

    # Thêm cell đóng gói ở cuối
    cells.append(md("---\n# ⏵ Đóng gói kết quả (Kaggle Artifacts)"))
    cells.append(code(ZIP_BUNDLE_CODE))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = render_kaggle()
    OUT_FILE.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote kaggle/Lab22_DPO_Kaggle.ipynb ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
