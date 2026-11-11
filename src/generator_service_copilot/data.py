from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

INTENTS = (
    "alarm_interpretation",
    "consumables",
    "electrical_inspection",
    "gas_quality",
    "human_escalation",
    "installation",
    "maintenance",
    "pressure_flow",
)

MVTEC_EXPECTED_COUNTS = {
    ("train", "good"): 224,
    ("test", "good"): 58,
    ("test", "bent_wire"): 13,
    ("test", "cable_swap"): 12,
    ("test", "combined"): 11,
    ("test", "cut_inner_insulation"): 14,
    ("test", "cut_outer_insulation"): 10,
    ("test", "missing_cable"): 12,
    ("test", "missing_wire"): 10,
    ("test", "poke_insulation"): 10,
}

SCENARIOS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "alarm_interpretation": (
        ("status display", "shows an unfamiliar warning after startup", False),
        ("alarm history", "contains a recurring sensor warning", False),
        ("control panel", "shows two warnings at the same time", False),
        ("indicator lamp", "remains amber after the checklist is complete", False),
        ("event log", "records an intermittent communication alert", False),
        ("display", "clears a warning that returns on the next cycle", False),
        ("controller", "reports a maintenance reminder", False),
        ("alarm banner", "appears without a matching operator note", False),
        ("status page", "reports a sensor value as unavailable", False),
        ("fault summary", "contains an unknown code that is not in the handover notes", False),
    ),
    "consumables": (
        ("filter element", "is approaching its documented replacement point", False),
        ("water-treatment cartridge", "has no recorded installation date", False),
        ("seal kit", "was opened but its storage condition is unknown", False),
        ("dryer material", "is listed as due for inspection", False),
        ("sample-line filter", "has visible contamination", False),
        ("approved lubricant", "cannot be matched to the service record", False),
        ("replacement cartridge", "has a different part label from the removed item", False),
        ("consumables log", "contains a missing batch identifier", False),
        ("spare filter", "has damaged packaging", False),
        ("service kit", "is incomplete before a scheduled visit", False),
    ),
    "electrical_inspection": (
        ("cable harness", "has a visible cut in its outer insulation", True),
        ("terminal block", "contains a connector that appears loose", True),
        ("cabinet cable", "is bent sharply near its entry point", True),
        ("protective earth lead", "has an unreadable inspection tag", True),
        ("sensor cable", "shows abrasion near a clamp", True),
        ("connector", "has discoloration around one contact", True),
        ("wiring channel", "contains an unsecured cable", True),
        ("cabinet gland", "does not appear to grip the cable evenly", True),
        ("power lead", "has a damaged outer jacket", True),
        ("inspection image", "shows a cable that differs from the reference layout", True),
    ),
    "gas_quality": (
        ("quality display", "shows a reading outside the expected operating state", True),
        ("sample line", "may not have been purged before a quality check", True),
        ("quality trend", "changes abruptly after maintenance", True),
        ("analyzer status", "is not marked ready during a product check", True),
        ("sampling record", "does not identify the sampling point", False),
        ("quality log", "contains a gap during a production run", False),
        ("product reading", "does not agree with the secondary instrument", True),
        ("sample connection", "was changed before an unexpected reading", True),
        ("quality certificate", "cannot be linked to the current service event", False),
        ("analyzer", "reports unstable values after its readiness check", True),
    ),
    "human_escalation": (
        ("equipment enclosure", "has a suspected gas leak", True),
        ("service area", "has an active fire or smoke condition", True),
        ("operator", "reports symptoms after a possible gas exposure", True),
        ("safety interlock", "was bypassed before the service request", True),
        ("pressure boundary", "has visible physical damage", True),
        ("ventilation system", "is unavailable in the equipment area", True),
        ("unknown contractor modification", "changes the approved equipment layout", True),
        ("emergency stop", "does not remain in its expected state", True),
        ("electrical cabinet", "has signs of overheating", True),
        ("site condition", "cannot be made safe using the approved local procedure", True),
    ),
    "installation": (
        ("installation record", "is missing the equipment identifier", False),
        ("handover checklist", "has not been signed after commissioning", False),
        ("equipment location", "does not match the approved layout drawing", True),
        ("ventilation checklist", "is incomplete before startup", True),
        ("utility connection", "has no recorded verification result", True),
        ("startup checklist", "contains an unresolved item", True),
        ("site acceptance record", "omits the responsible reviewer", False),
        ("equipment label", "does not match the handover document", False),
        ("installation photo set", "is missing the cabinet entry view", False),
        ("operator handover", "does not include the escalation contacts", False),
    ),
    "maintenance": (
        ("preventive-maintenance schedule", "shows an overdue inspection", False),
        ("service log", "does not identify the completed checklist", False),
        ("inspection record", "contains an unresolved observation", False),
        ("maintenance counter", "does not match the written service date", False),
        ("post-service check", "was not attached to the work order", False),
        ("cleaning record", "does not state which approved method was used", False),
        ("maintenance note", "does not identify the replaced component", False),
        ("return-to-service record", "is missing reviewer approval", False),
        ("routine inspection", "found a condition not covered by the checklist", True),
        ("service interval", "cannot be confirmed from the available records", False),
    ),
    "pressure_flow": (
        ("outlet flow indication", "drops during otherwise steady operation", True),
        ("pressure trend", "changes after a service activity", True),
        ("flow display", "does not agree with the secondary reading", True),
        ("outlet reading", "is unstable during the documented check", True),
        ("pressure record", "contains a sudden step change", True),
        ("flow trend", "shows repeated short interruptions", True),
        ("instrument status", "is not ready when a pressure reading is recorded", True),
        ("process connection", "was changed before a low-flow report", True),
        ("trend log", "has a gap around an unexpected pressure event", False),
        ("reported flow issue", "cannot be reproduced using the approved checklist", False),
    ),
}

PARAPHRASES = (
    "The {component} {symptom}. What documented service path applies?",
    "A customer reports that the {component} {symptom}. Route this request safely.",
    (
        "During a routine visit, the technician found that the {component} {symptom}. "
        "Which procedure should be reviewed?"
    ),
    "Service request: the {component} {symptom}. Identify the next evidence-based step.",
)

CASE_SCHEMA = DataFrameSchema(
    {
        "case_id": Column(str, Check.str_length(min_value=1), unique=True),
        "group_id": Column(str, Check.str_length(min_value=1)),
        "text": Column(str, Check.str_length(min_value=20)),
        "intent": Column(str, Check.isin(INTENTS)),
        "component": Column(str, Check.str_length(min_value=1)),
        "safety_critical": Column(bool),
        "synthetic": Column(bool, Check.eq(True)),
        "split": Column(str, Check.isin(("train", "development", "test"))),
    },
    strict=True,
)

CHUNK_SCHEMA = DataFrameSchema(
    {
        "chunk_id": Column(str, Check.str_length(min_value=12), unique=True),
        "document_id": Column(str, Check.str_length(min_value=1)),
        "title": Column(str, Check.str_length(min_value=1)),
        "section": Column(str, Check.str_length(min_value=1)),
        "text": Column(str, Check.str_length(min_value=30)),
        "source_file": Column(str, Check.str_matches(r".+\.md$")),
        "source_version": Column(str, Check.str_length(min_value=1)),
        "equipment_family": Column(str, Check.str_length(min_value=1)),
        "safety_topic": Column(str, Check.str_length(min_value=1)),
        "synthetic": Column(bool, Check.eq(True)),
    },
    strict=True,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_groups(intent: str, count: int) -> dict[int, str]:
    ordered = sorted(
        range(count),
        key=lambda index: _sha256_bytes(f"{intent}:{index}".encode()),
    )
    return {
        index: "train" if rank < 6 else "development" if rank < 8 else "test"
        for rank, index in enumerate(ordered)
    }


def generate_service_cases() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for intent, scenarios in SCENARIOS.items():
        split_by_index = _split_groups(intent, len(scenarios))
        for scenario_index, (component, symptom, safety_critical) in enumerate(scenarios):
            group_id = f"{intent}-{scenario_index:02d}"
            for variant, template in enumerate(PARAPHRASES):
                records.append(
                    {
                        "case_id": f"{group_id}-v{variant + 1}",
                        "group_id": group_id,
                        "text": template.format(component=component, symptom=symptom),
                        "intent": intent,
                        "component": component,
                        "safety_critical": safety_critical,
                        "synthetic": True,
                        "split": split_by_index[scenario_index],
                    }
                )
    cases = pd.DataFrame.from_records(records)
    return validate_service_cases(cases)


def validate_service_cases(cases: pd.DataFrame) -> pd.DataFrame:
    validated = CASE_SCHEMA.validate(cases)
    leakage = validated.groupby("group_id")["split"].nunique()
    if (leakage != 1).any():
        raise ValueError("A semantic scenario group crosses data splits")
    class_presence = validated.groupby(["split", "intent"]).size().unstack(fill_value=0)
    if (class_presence == 0).any().any() or set(class_presence.columns) != set(INTENTS):
        raise ValueError("Every intent must be represented in every split")
    normalized = validated["text"].map(lambda value: re.sub(r"\W+", " ", value.lower()).strip())
    duplicate_splits = (
        validated.assign(normalized=normalized).groupby("normalized")["split"].nunique()
    )
    if (duplicate_splits > 1).any():
        raise ValueError("Normalized duplicate text crosses data splits")
    return validated.sort_values("case_id").reset_index(drop=True)


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        raise ValueError("Manual is missing YAML-style front matter")
    _, metadata_text, body = text.split("---\n", 2)
    metadata: dict[str, str] = {}
    for line in metadata_text.splitlines():
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    required = {
        "document_id",
        "title",
        "version",
        "equipment_family",
        "safety_topic",
        "fictional",
    }
    if set(metadata) != required or metadata["fictional"].lower() != "true":
        raise ValueError("Manual metadata is incomplete or not marked fictional")
    return metadata, body.strip()


def _append_manual_chunk(
    records: list[dict[str, object]],
    metadata: dict[str, str],
    section: str,
    paragraphs: list[str],
    source_file: str,
) -> None:
    if not paragraphs:
        return
    content = "\n\n".join(paragraphs).strip()
    identity = f"{metadata['document_id']}:{metadata['version']}:{section}:{content}"
    records.append(
        {
            "chunk_id": _sha256_bytes(identity.encode())[:16],
            "document_id": metadata["document_id"],
            "title": metadata["title"],
            "section": section,
            "text": content,
            "source_file": source_file,
            "source_version": metadata["version"],
            "equipment_family": metadata["equipment_family"],
            "safety_topic": metadata["safety_topic"],
            "synthetic": True,
        }
    )


def load_document_chunks(manual_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(manual_dir.glob("*.md")):
        metadata, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        current_section = metadata["title"]
        paragraphs: list[str] = []

        for block in re.split(r"\n\s*\n", body):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                _append_manual_chunk(records, metadata, current_section, paragraphs, path.name)
                paragraphs.clear()
                current_section = block.lstrip("#").strip()
            else:
                paragraphs.append(re.sub(r"\s+", " ", block))
        _append_manual_chunk(records, metadata, current_section, paragraphs, path.name)

    if not records:
        raise ValueError(f"No manual documents found in {manual_dir}")
    return CHUNK_SCHEMA.validate(pd.DataFrame.from_records(records)).sort_values("chunk_id")


def image_manifest(mvtec_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    image_paths = [
        *mvtec_root.glob("cable/train/*/*.*"),
        *mvtec_root.glob("cable/test/*/*.*"),
    ]
    for image_path in sorted(image_paths):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        relative = image_path.relative_to(mvtec_root)
        parts = relative.parts
        if "ground_truth" in parts:
            continue
        dataset_split = "train" if "train" in parts else "test"
        defect_type = image_path.parent.name
        mask_path = ""
        if dataset_split == "test" and defect_type != "good":
            candidate = (
                mvtec_root / "cable" / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
            )
            if candidate.exists():
                mask_path = candidate.relative_to(mvtec_root).as_posix()
        records.append(
            {
                "path": relative.as_posix(),
                "dataset_split": dataset_split,
                "defect_type": defect_type,
                "is_anomaly": defect_type != "good",
                "mask_path": mask_path,
                "sha256": sha256_file(image_path),
                "bytes": image_path.stat().st_size,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "path",
                "dataset_split",
                "project_split",
                "defect_type",
                "is_anomaly",
                "mask_path",
                "sha256",
                "bytes",
            ]
        )
    manifest = pd.DataFrame.from_records(records).drop_duplicates(subset="path")
    if manifest["path"].duplicated().any():
        raise ValueError("Image paths are not unique")
    manifest = _assign_mvtec_project_splits(manifest)
    validate_mvtec_snapshot(manifest, mvtec_root)
    return manifest


def _assign_mvtec_project_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    result["project_split"] = ""

    official_train = result.index[result["dataset_split"] == "train"].tolist()
    ordered_train = sorted(
        official_train,
        key=lambda index: _sha256_bytes(result.loc[index, "path"].encode()),
    )
    train_cutoff = int(len(ordered_train) * 0.8)
    result.loc[ordered_train[:train_cutoff], "project_split"] = "train"
    result.loc[ordered_train[train_cutoff:], "project_split"] = "development"

    official_test = result.loc[result["dataset_split"] == "test"]
    for _, group in official_test.groupby("defect_type"):
        ordered = sorted(
            group.index,
            key=lambda index: _sha256_bytes(result.loc[index, "path"].encode()),
        )
        development_count = (len(ordered) + 1) // 2
        result.loc[ordered[:development_count], "project_split"] = "development"
        result.loc[ordered[development_count:], "project_split"] = "test"

    if (result["project_split"] == "").any():
        raise ValueError("Every MVTec image must have a project split")
    return result.sort_values("path").reset_index(drop=True)


def validate_mvtec_snapshot(manifest: pd.DataFrame, mvtec_root: Path) -> None:
    counts = manifest.groupby(["dataset_split", "defect_type"]).size().to_dict()
    if counts != MVTEC_EXPECTED_COUNTS:
        raise ValueError(f"MVTec cable snapshot is incomplete or unexpected: {counts}")
    anomalies = manifest.loc[manifest["is_anomaly"]]
    if len(anomalies) != 92 or (anomalies["mask_path"] == "").any():
        raise ValueError("Every anomalous MVTec cable image must have a mask")
    if manifest["sha256"].nunique() != len(manifest):
        raise ValueError("MVTec cable snapshot contains duplicate image content")
    if set(manifest["project_split"]) != {"train", "development", "test"}:
        raise ValueError("MVTec project splits are incomplete")
    final_test = manifest.loc[manifest["project_split"] == "test"]
    if set(final_test["defect_type"]) != set(
        manifest.loc[manifest["dataset_split"] == "test", "defect_type"]
    ):
        raise ValueError(
            "Every official test category must be represented in the sealed test split"
        )
    license_path = mvtec_root / "cable" / "license.txt"
    if not license_path.exists() or sha256_file(license_path) != (
        "a355a57937552790ebd751f8a8026b23882160ab73f1c38f8030456dfcf1a330"
    ):
        raise ValueError("MVTec license is missing or has an unexpected checksum")


def _file_manifest(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _write_eda(cases: pd.DataFrame, chunks: pd.DataFrame, artifact_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(artifact_dir.parents[1] / ".mplconfig"))
    from matplotlib.figure import Figure

    artifact_dir.mkdir(parents=True, exist_ok=True)

    counts = cases.groupby(["intent", "split"]).size().unstack(fill_value=0)
    figure = Figure(figsize=(11, 6), constrained_layout=True)
    axis = figure.subplots()
    counts.plot(kind="bar", stacked=True, ax=axis, color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set_title("Synthetic service cases by intent and split")
    axis.set_xlabel("Intent")
    axis.set_ylabel("Cases")
    axis.tick_params(axis="x", rotation=35)
    figure.savefig(artifact_dir / "case_balance.png", dpi=160)

    lengths = chunks.assign(words=chunks["text"].str.split().str.len()).sort_values("words")
    figure = Figure(figsize=(10, 5), constrained_layout=True)
    axis = figure.subplots()
    axis.barh(lengths["section"], lengths["words"], color="#4C78A8")
    axis.set_title("Fictional manual chunk lengths")
    axis.set_xlabel("Words")
    figure.savefig(artifact_dir / "document_chunk_lengths.png", dpi=160)


def prepare_data(project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    source_dir = project_root / "data" / "source"
    processed_dir = project_root / "data" / "processed"
    artifact_dir = project_root / "artifacts" / "eda"
    processed_dir.mkdir(parents=True, exist_ok=True)

    cases = generate_service_cases()
    chunks = load_document_chunks(source_dir / "manuals")
    cases.to_csv(processed_dir / "service_cases.csv", index=False)
    for split in ("train", "development", "test"):
        cases.loc[cases["split"] == split].to_csv(
            processed_dir / f"service_cases_{split}.csv", index=False
        )
    chunks.to_csv(processed_dir / "document_chunks.csv", index=False)

    images = image_manifest(project_root / "data" / "external" / "mvtec_ad")
    images.to_csv(processed_dir / "mvtec_cable_manifest.csv", index=False)
    _write_eda(cases, chunks, artifact_dir)

    case_counts = cases.groupby(["split", "intent"]).size().unstack(fill_value=0)
    summary: dict[str, object] = {
        "service_cases": {
            "rows": len(cases),
            "semantic_groups": int(cases["group_id"].nunique()),
            "intents": int(cases["intent"].nunique()),
            "safety_critical_rows": int(cases["safety_critical"].sum()),
            "split_rows": cases["split"].value_counts().sort_index().to_dict(),
            "rows_per_split_and_intent": {
                split: {intent: int(value) for intent, value in values.items()}
                for split, values in case_counts.iterrows()
            },
            "synthetic": True,
        },
        "documents": {
            "source_documents": int(chunks["document_id"].nunique()),
            "chunks": len(chunks),
            "versions": sorted(chunks["source_version"].unique().tolist()),
            "synthetic": True,
        },
        "mvtec_cable": {
            "status": "available" if len(images) else "not_downloaded",
            "images": len(images),
            "train_normal_images": int(
                ((images["dataset_split"] == "train") & ~images["is_anomaly"]).sum()
            )
            if len(images)
            else 0,
            "test_normal_images": int(
                ((images["dataset_split"] == "test") & ~images["is_anomaly"]).sum()
            )
            if len(images)
            else 0,
            "test_anomaly_images": int(images["is_anomaly"].sum()) if len(images) else 0,
            "anomaly_masks": int((images["mask_path"] != "").sum()) if len(images) else 0,
            "project_split_rows": images["project_split"].value_counts().sort_index().to_dict()
            if len(images)
            else {},
            "bytes": int(images["bytes"].sum()) if len(images) else 0,
            "license": "CC BY-NC-SA 4.0",
        },
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    tracked_paths = [
        *source_dir.rglob("*.md"),
        project_root / "data" / "sources.json",
        *processed_dir.glob("*.csv"),
        artifact_dir / "summary.json",
    ]
    manifest = {"files": _file_manifest(tracked_paths, project_root)}
    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate project data")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.project_root), indent=2))


if __name__ == "__main__":
    main()
