# REALTYAI2 — WHOLE-PROJECT HYGIENE & CLEANUP REPORT

**Date**: October 2, 2026  
**Auditor**: RealtyAI Research Validation Committee  
**Evaluation Scope**: Whole-Project File Audit, Safe Deletion Execution & Repository Cleanliness  

---

## 1. File Count & Inventory Summary

- **Total Directories Audited**: 12 root-level directories + all subdirectories
- **Folders Removed**: 1 (`scratch/`) + transient compiler cache folders (`__pycache__/` across all modules)
- **Files Removed**: 9 temporary exploratory script files
- **Total Operational Files Retained**: Complete active codebase, all canonical checkpoints, manifests, preprocessed datasets, tests, and documentation intact.

---

## 2. Inventory of Removed Items & Safe-Deletion Justifications

Under the project's strict 13-point Safe File-Deletion Rule, an item may only be removed if it is not required by any active runtime path, test, pipeline, model inference, reproducibility workflow, documentation, or historical research evidence.

| Removed Path | Type | File Size | Safe Deletion Justification |
| :--- | :--- | :---: | :--- |
| `scratch/check_cache_and_master.py` | Python Script | 866 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role. |
| `scratch/check_metrics.py` | Python Script | 1,018 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role. |
| `scratch/check_order_stat.py` | Python Script | 3,117 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role (conformal order statistic calculation is formally verified in `scripts/final_independent_verification.py` and unit-tested in `tests/test_models.py`). |
| `scratch/check_peer_splits.py` | Python Script | 874 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role (split verification is part of `tests/test_data_pipeline.py`). |
| `scratch/check_zillow_combinations.py` | Python Script | 843 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role. |
| `scratch/freeze_artifacts.py` | Python Script | 1,600 B | Deleted because it was a temporary exploratory artifact, had no active references, was not needed for reproducibility/provenance, and had no archival role (artifact hashes are verified in `scripts/final_independent_verification.py`). |
| `scratch/independent_peer.py` | Python Script | 3,883 B | Deleted because it was a temporary exploratory artifact; its logic was fully consolidated into canonical `scripts/final_independent_verification.py`. |
| `scratch/independent_spacenet.py` | Python Script | 5,466 B | Deleted because it was a temporary exploratory artifact; its logic was fully consolidated into canonical `scripts/final_independent_verification.py`. |
| `scratch/independent_zillow.py` | Python Script | 4,086 B | Deleted because it was a temporary exploratory artifact; its logic was fully consolidated into canonical `scripts/final_independent_verification.py`. |
| `scratch/` | Directory | — | Deleted because all contained files were temporary exploratory artifacts that were removed. |
| `**/__pycache__/` | Directories | Variable | Deleted because they are transient Python bytecode caches that are automatically regenerated at runtime and have no reproducibility value. |

---

## 3. Duplicate File Audit

| File Group | Identified Paths | Hash / Equivalence Analysis | Retention Decision & Justification |
| :--- | :--- | :--- | :--- |
| **SpaceNet U-Net Model** | `models/saved/unet_spacenet_v1.pt`<br>`models/saved/unet_satellite.pt` | `unet_spacenet_v1.pt` is canonical (SHA-256: `28bedc6a...`). `unet_satellite.pt` has 0 weight differences across all 118 tensors. | **KEEP BOTH**. Canonical file is strictly loaded by production code. Alias is preserved for historical test backward compatibility as documented in `docs/DECISIONS.md`. |
| **PEER ResNet Model** | `models/saved/resnet_peer_collapse_v1.pt`<br>`models/saved/resnet_condition.pt` | `resnet_peer_collapse_v1.pt` is canonical (SHA-256: `b5c5d063...`). `resnet_condition.pt` has 0 weight differences across all 129 tensors. | **KEEP BOTH**. Canonical file is strictly loaded by production code. Alias is preserved for historical test backward compatibility. |
| **Ames XGBoost Model** | `models/saved/xgboost_ames_v1.pkl`<br>`models/saved/xgboost_price.pkl` | Both files are byte-for-byte identical (SHA-256: `96b9ebc3...`). | **KEEP BOTH**. Canonical file is loaded by production code. Alias is preserved for historical test backward compatibility. |
| **Executive PDF Reports** | `RealtyAI_Executive_Project_Report.pdf`<br>`RealtyAI_Project_Summary_Report.pdf` | Byte-for-byte identical (SHA-256: `b96af247...`). Created by `generate_pdf_report.py` via `shutil.copyfile`. | **KEEP BOTH**. Both filenames are referenced in historical documentation and audit logs (`docs/VALIDATION_LOG.md`). Retaining both prevents broken links. |

---

## 4. Archival & Historical Materials Intentionally Retained

The following assets were audited and deliberately preserved to guarantee full academic reproducibility and experimental provenance:

1. **`models/legacy_synthetic/`**:
   - Contains legacy synthetic-trained checkpoints and metric JSONs from Phase 0.
   - Retained as explicit evidence of the project's migration from synthetic to authentic data, with a dedicated README documenting quarantine rationale.
2. **`data/synthetic_backup/`**:
   - Quarantined synthetic drawings and tiles from Phase 0.
   - Retained because it documents the historical development and ensures full auditability of the Phase 1 migration.
3. **`data/manifests/spacenet_smoke_test_manifest.csv` & `data/processed/spacenet_smoke_test_splits.json`**:
   - Retained because they define the isolated 20-chip CI smoke test partition used for automated regression tests.
4. **Historical Audit Documents (`docs/PHASE0_AUDIT.md`, `docs/PHASE4_PRECHECK.md`, `docs/PHASE4_FINAL_AUDIT.md`)**:
   - Retained because they provide the chronological provenance trail required by academic committees.

---

## 5. Canonical Artifacts Frozen & Retained

All canonical artifacts were verified to be present and unchanged:
- **`models/saved/unet_spacenet_v1.pt`** (7,821,487 bytes, SHA-256: `28bedc6a4750f99bdb2ef8ee2312d85f53d2b72fdf805c2a29e13db6bd4352f7`)
- **`models/saved/resnet_peer_collapse_v1.pt`** (45,056,045 bytes, SHA-256: `b5c5d063f1dfa2a040ad2f3e3b942e37bbf6d8da67f1399f34769ba127eebc69`)
- **`models/saved/xgboost_ames_v1.pkl`** (727,758 bytes, SHA-256: `96b9ebc30c5f2edcceedf4f422cc805f77aa979eae6a1a090decf3500780300e`)
- **`models/saved/forecaster_summary.json`** (13,052 bytes, SHA-256: `1d008e35b24bc80a4b424b1dfc03698a5844fb63d92a5331955cc56ab2ad0ecd`)
- **`models/saved/final_verification.json`** (8,988 bytes, generated by `scripts/final_independent_verification.py`)

---

## 6. Dependency & Portability Cleanliness

- **Dependencies (`requirements.txt`)**: Cleanly pinned with exact compatible versions. Added `reportlab>=4.0.0` for PDF generation completeness.
- **Portability Audit**: Zero active absolute Windows paths (`C:\Users\...`) exist in runtime Python files, JSON configs, or CSV manifests. All dataset manifests use strictly relative paths (`data/raw/...`).

---

## 7. Hygiene Audit Conclusion

The repository is certified clean, minimal, uncluttered, and free of orphaned exploratory scripts, while 100% of required source files, canonical models, processed arrays, tests, documentation, and historical provenance materials are preserved intact.
