# MachineLearning

Repository to share progress and to manage versions of exam MachineLearning (M14)

Deliverables: 
- The final notebook is in `ExamDataMining_Final.ipynb`
- The executive summary is in : `Executive sumary.docx`
- The final list of selected clients is in : `output/selected_guests.csv`

Besides that : 

- `funcs`: contains some common functions
- `input`: contains the input data
- `models`: contains cached version of model optimization scans for fast re-loading

The **playground notebooks** were moved to the `playground` subfolder after finalization

Some discussions between the team members are available under the github issues : https://github.com/Marijkevandesteene/MachineLearning/issues?q=is%3Aissue

A brief changelog is included in the preamble of the final notebook, which is repeated here : 

## Changelog

- **2024-06-05** [MV] : Initial version
- **2024-06-06** [BM] : Consolidated structure, imported initial analysis from notebooks 
- **2024-06-18** [BM] : Consolidated structure: Walkthrough in team / Finalized data preparation
- **2024-06-18** [MVDS] : Added data preparation steps on score
- **2024-06-19** [BM] : Fixed issue w.r.t kNN imputer to apply for score
- **2024-06-23** [MVDS] : Random forest model tuning, Calibration / applied explainability / removed some try out code
- **2024-06-25** [DL] : Exploring SVM 
- **2024-06-28** [BM] : Added Gradient Booster + consolidation of notebook
- **2024-06-29** [MVDS] : White boxing for selected revenue_best model (GBM) / selection of hotel guest / issues
- **2024-06-30** [BM] : Final cleanup, added more interpretation on revenu & correlation analysis & generated html
