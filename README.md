## Purpose

Here, we evaluate if a commercial LLM can better represent shape images compared to our slot-based CNN model. The results are available in this directory is evaluated against the same test set and the same slot accuracy metric as used by the CNN models.
## Files

- `run_llm_eval.py` — python script that takes labels CSV + image folder, and sends each image to
  **Google Gemini** via the `google-genai` 

- `run_100/predictions.csv` — per-image gold + prediction + slot-match flags
  for the run 
  
- `run_100/metrics.json` — aggregate slot/relation/attribute/exact accuracy
  for the run
  
- `run_100/sample_index.csv` — the 100 (image_id, file_name, gold) rows that
  were sampled so anyone can re-run on the identical sample.


## Re-running

```bash
pip install google-genai pandas scikit-learn
export GEMINI_API_KEY=...
cd llm_comparison
python run_llm_eval.py \
    --labels_csv ../labels.csv \
    --image_dir  ../images \
    --n_samples  200 \
    --model      gemini-2.5-flash \
    --out_dir    ./run_200
```

A Gemini API key can be created free of charge at
https://aistudio.google.com/apikey . The `gemini-3.1-Flash-Lite-Preview` (500 requests/day).
