# Original Source Map

This repository keeps the main reproducible workflow in `src/dbrw_extratrees_pipeline.py`.
The files below are preserved under `original_sources/` so the original development
history remains understandable.

Note: some original scripts contain the local paths that existed on the working
PC. They are preserved as historical source files, not as the recommended entry
point. The consolidated script uses command-line paths instead.

| Original file | Role in this repo |
| --- | --- |
| `make_features.py` | Raw multimodal lifelog aggregation into daily and evening-window feature tables. Kept as source reference because raw competition data are not redistributed. |
| `19_make_sleep_window_features.py` | Extension script that added additional sleep-window feature tables. |
| `20_add_sleep_window_multiday_features.py` | Extension script that added rolling, delta, and subject z-score features to sleep-window features. |
| `compare_window_features.py` | Main source for the DBRW window ablation and the `0.6575` vs `0.6590` validation result. Consolidated into `validate-windows`. |
| `03_make_submission_extratrees_shrinkage.py` | Main source for final ExtraTrees shrinkage submission generation. Consolidated into `make-submission`. |
| `step1_train.py` | Earlier paper-support training run using ExtraTrees and OOB-style shrinkage. |
| `step2_feature_importance.py` | Source for ExtraTrees feature-importance extraction. Consolidated into `feature-importance`. |
| `step3_shap.py` | Earlier SHAP visualization script. Preserved for paper interpretation history. |
| `step4_figures_v2.py` | Source for Word-paper figures. A compact version is consolidated into `make-readme-figures`. |
| `step5_tables.py` | Source for paper table CSV generation. |
| `step6_paper_final.py` | Source for the last lightweight Word draft generator. |
| `paper_generate_assets.py` | Source for the LaTeX/ICTC paper assets under `paper_assets/`. |
| `200_final_research_multiview_pipeline.py` | Later exploratory multiview pipeline; kept separately because it is broader than the final DBRW ExtraTrees result in the submitted Word paper. |
