#!/usr/bin/env python3
"""Build a balanced single-label OEE dataset from the existing multi-label CSV.

Strategy:
- Read the current dataset with 4 binary labels.
- Infer a dominant label from question keywords.
- Keep only rows with a clear dominant label.
- Export a one-hot encoded dataset with equal rows per class.
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

INPUT_PATH = Path("/home/emanosalvasb/PUBLICO/DEEP LEARNING/usfq-mmia-main/data/OEE_dataset_clasificado_1.csv")
OUTPUT_PATH = Path("/home/emanosalvasb/PUBLICO/DEEP LEARNING/usfq-mmia-main/data/OEE_dataset_clasificado_balanced_single_label.csv")
SEED = 42

LABELS = ["calidad", "velocidad", "rendimiento", "eficiencia_OEE"]

KEYWORDS = {
    "calidad": [
        "calidad", "defecto", "rechazo", "ppm", "retraba", "inspecci", "mala calidad", "copq"
    ],
    "velocidad": [
        "velocidad", "ciclo", "tasa de produ", "capacidad nominal", "cuello de botella", "ritmo"
    ],
    "rendimiento": [
        "rendimiento", "parada", "tiempo muerto", "disponibilidad", "aver", "falla", "downtime"
    ],
    "eficiencia_OEE": [
        "oee", "eficiencia", "efectividad", "global", "indicador", "componente"
    ],
}


def normalize_text(text: str) -> str:
    return text.lower().strip()


def score_labels(question: str) -> Dict[str, int]:
    q = normalize_text(question)
    scores = {label: 0 for label in LABELS}
    for label, words in KEYWORDS.items():
        for w in words:
            if re.search(re.escape(w), q):
                scores[label] += 1
    return scores


def choose_label(question: str) -> str | None:
    scores = score_labels(question)
    best = max(scores.values())
    if best == 0:
        return None
    winners = [label for label, score in scores.items() if score == best]
    if len(winners) == 1:
        return winners[0]

    # Tie-breaker by first mention in text.
    q = normalize_text(question)
    first_pos: List[Tuple[int, str]] = []
    for label in winners:
        positions = [q.find(k) for k in KEYWORDS[label] if q.find(k) >= 0]
        if positions:
            first_pos.append((min(positions), label))
    if not first_pos:
        return None
    first_pos.sort(key=lambda x: x[0])
    return first_pos[0][1]


def read_rows() -> List[Dict[str, str]]:
    with INPUT_PATH.open("r", encoding="latin1", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_balanced(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    dropped = 0

    for row in rows:
        question = row.get("pregunta", "").strip()
        if not question:
            dropped += 1
            continue

        label = choose_label(question)
        if label is None:
            dropped += 1
            continue

        out = {"pregunta": question}
        for l in LABELS:
            out[l] = "1" if l == label else "0"
        by_label[label].append(out)

    min_count = min(len(by_label[l]) for l in LABELS)
    rng = random.Random(SEED)

    balanced: List[Dict[str, str]] = []
    for label in LABELS:
        sample = by_label[label][:]
        rng.shuffle(sample)
        balanced.extend(sample[:min_count])

    rng.shuffle(balanced)

    print("Input rows:", len(rows))
    print("Dropped rows (no clear dominant label):", dropped)
    print("Rows per class before balancing:")
    for label in LABELS:
        print(f"  {label:<15} {len(by_label[label])}")
    print("Balanced rows per class:", min_count)
    print("Output rows:", len(balanced))

    return balanced


def write_rows(rows: List[Dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pregunta", *LABELS])
        writer.writeheader()
        writer.writerows(rows)


def sanity_check(rows: List[Dict[str, str]]) -> None:
    counts = {l: 0 for l in LABELS}
    for row in rows:
        active = sum(int(row[l]) for l in LABELS)
        if active != 1:
            raise ValueError(f"Invalid row with {active} active labels: {row}")
        for l in LABELS:
            counts[l] += int(row[l])

    print("Sanity check passed. Label counts:")
    for l in LABELS:
        print(f"  {l:<15} {counts[l]}")


def main() -> None:
    rows = read_rows()
    balanced = build_balanced(rows)
    write_rows(balanced)
    sanity_check(balanced)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
