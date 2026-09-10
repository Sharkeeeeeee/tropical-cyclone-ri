"""Check that numbers quoted in the reports match the files they came from.

Reports are written by hand from console output, so they drift: a rerun changes
a value and the prose keeps the old one. This re-derives every headline figure
from the saved CSVs and compares.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import summary  # noqa: E402

fails = []


def check(label, quoted, actual, tol=5e-4):
    ok = abs(quoted - actual) <= tol
    print(f"[{'  ok  ' if ok else ' FAIL '}] {label:<52} "
          f"report {quoted:.4f}  actual {actual:.4f}")
    if not ok:
        fails.append(f"{label}: report {quoted:.4f}, actual {actual:.4f}")


def quoted_in(path, pattern, group=1):
    """Pull a number out of a report so the comparison uses the real text."""
    txt = (REPORTS_DIR / path).read_text(encoding="utf-8")
    m = re.search(pattern, txt)
    if not m:
        fails.append(f"{path}: pattern not found -- {pattern}")
        return float("nan")
    return float(m.group(group))


def main():
    print("=" * 74)
    print("report figures vs source files")
    print("=" * 74)

    # ---- structure_report.md against the OOF predictions it describes
    sof = pd.read_csv(PROCESSED_DIR / "structure_oof_recon_2000_2022.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    y = sof["ri"].to_numpy(int)
    print(f"\nstructure_oof: {len(sof):,} rows, {y.sum():,} RI "
          f"({100*y.mean():.2f}%)")
    for col, label, pat in [
        ("p_environment", "environment PR AUC",
         r"\| environment \| 8 \| [\d.]+ \| ([\d.]+) \|"),
        ("p_environment_plus_SHIPS_IR", "env + SHIPS IR PR AUC",
         r"\| environment \+ SHIPS IR \| 10 \| [\d.]+ \| ([\d.]+) \|"),
        ("p_environment_plus_structure", "env + structure PR AUC",
         r"\| \*\*environment \+ structure\*\* \| 149 \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\*"),
        ("p_structure_only", "structure only PR AUC",
         r"\| structure only \| 141 \| [\d.]+ \| ([\d.]+) \|"),
    ]:
        check(label, quoted_in("structure_report.md", pat),
              summary(y, sof[col].to_numpy(), "")["PR_AUC"])

    # the headline ratio
    env = summary(y, sof["p_environment"].to_numpy(), "")["PR_AUC"]
    ir = summary(y, sof["p_environment_plus_SHIPS_IR"].to_numpy(), "")["PR_AUC"]
    st = summary(y, sof["p_environment_plus_structure"].to_numpy(), "")["PR_AUC"]
    ratio = (st - env) / (ir - env)
    check("structure / SHIPS-IR gain ratio",
          quoted_in("structure_report.md", r"worth ([\d.]+)× the classical pair"),
          ratio, tol=0.02)

    # ---- CNN comparison
    cnn = pd.read_csv(PROCESSED_DIR / "cnn_oof_recon_2000_2022_seed0.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    j = sof.merge(cnn[["SID", "ISO_TIME", "p_cnn"]], on=["SID", "ISO_TIME"])
    yj = j["ri"].to_numpy(int)
    check("CNN only PR AUC",
          quoted_in("structure_report.md",
                    r"\| CNN only \(391k parameters\) \| [\d.]+ \| ([\d.]+) \|"),
          summary(yj, j["p_cnn"].to_numpy(), "")["PR_AUC"])
    check("structure only PR AUC (CNN table)",
          quoted_in("structure_report.md",
                    r"\| \*\*structure only\*\* \(141 features\) \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\*"),
          summary(yj, j["p_structure_only"].to_numpy(), "")["PR_AUC"])

    # ---- experiment A transfer numbers
    tr = pd.read_csv(REPORTS_DIR / "expA_transfer.csv").set_index("model")
    for key, pat in [
        ("environment: recon -> recon",
         r"\| environment: recon → recon \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|"),
        ("environment: recon -> Dvorak",
         r"\| environment: recon → Dvorak \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|"),
    ]:
        check(key, quoted_in("experiment_a_report.md", pat),
              float(tr.loc[key, "ROC_AUC"]))

    # ---- baseline report
    base = pd.read_csv(REPORTS_DIR / "baseline_summary.csv").set_index("model")
    check("GBM env + IR PR AUC (baseline report)",
          quoted_in("baseline_report.md",
                    r"\| \*\*GBM, environment \+ IR\*\* \| 93 \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\*"),
          float(base.loc["GBM env + IR", "PR_AUC"]))

    # ---- quantization
    q = pd.read_csv(REPORTS_DIR / "expA_quantization.csv").set_index("sample")
    check("recon quantization contrast",
          quoted_in("experiment_a_report.md",
                    r"\| Recon-verified \(NA/EP\) \| [\d,]+ \| \*\*\+([\d.]+)\*\*"),
          float(q.loc["recon-verified (NA/EP)", "contrast"]))
    check("Dvorak quantization contrast",
          quoted_in("experiment_a_report.md",
                    r"\| Dvorak-derived \(all others\) \| [\d,]+ \| \*\*\+([\d.]+)\*\*"),
          float(q.loc["Dvorak-derived (others)", "contrast"]))

    print("\n" + "=" * 74)
    if fails:
        print(f"{len(fails)} mismatch(es):")
        for f in fails:
            print("  " + f)
        return 1
    print("all quoted figures match their source files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
