# gray-swan-alignment
A Gray Swan technical: **DPO-based alignment**.

---

## Repository Structure

- **data/**  
  Contains raw and synthetic evaluation datasets.  
- **notebooks/**  
  Contains Jupyter notebooks (e.g., `demo.ipynb`) demonstrating experiments end-to-end.  
- **src/gray_swan/**  
  The main Python package divided into modules for configuration, data processing, training, evaluation, and interpretability.  
- `__init__.py`  
  Placed in each subdirectory to mark them as Python packages, enabling imports using package paths (e.g., `from gray_swan.evaluation import harmful_eval`).

---

## Setup

### 1) Create Conda Environment & Activate It

```bash
conda create -n gray-swan python=3.10.13
conda activate gray-swan

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2) DeepSeek Configuration

We use **deepseek-chat** to generate synthetic datasets and classify helpfulness.  
Please fill in your API key in the file:
```
src/gray_swan/config/deepseek_config.yaml
```

**Example content:**

```yaml
api_key: "YOUR_API_KEY_HERE"
base_url: "https://api.deepseek.com"
model_name: "deepseek-chat"
```

### 3) Baseline Model

For all experiments, our baseline model is:
```
HuggingFaceTB/SmolLM2-135M-Instruct
```

---

## Example Usage

### Synthetic Data Generation
To generate synthetic completions for your dataset:

```bash
python src/gray_swan/data_preprocessing/synthetic_generation.py
```

### DPO + Utility Training
Run DPO training (with added utility loss):

```bash
python src/gray_swan/dpo_training/utility_dpo_batch_trainer.py
```

### Synthetic Evaluation
Generate completions for both the original and post-trained models on the evaluation set:

```bash
python src/gray_swan/evaluation/synthetic_eval_generation.py
```

### Harmfulness Evaluation
Evaluate harmful completions using the **HarmBench** classifier:

```bash
python src/gray_swan/evaluation/harmful_eval.py
```

### Utility (Helpfulness) Evaluation
Evaluate the helpfulness of completions via **DeepSeek**:

```bash
python src/gray_swan/evaluation/utility_eval.py
```

---

## Token Attribution

Token attribution code is available at:
```
src/gray_swan/interpretability/attribution_analyzer.py
```

Run it with:

```bash
python src/gray_swan/interpretability/attribution_analyzer.py
```

Alternatively, for a nicer looking in-notebook visualization:

```python
from src.gray_swan.interpretability.attribution_analyzer import AttributionAnalyzer

comparison_path = "/home/davidh/gray-swan-alignment/5_example_model_comparison.json"
orig_ckpt = "/data1/shared_models/SmolLM2-135M-Instruct"
tuned_ckpt = "/home/davidh/gray-swan-alignment/src/gray_swan/dpo_training/models/dpo_finetuned_grad_accum"
device = "cuda"

analyzer = AttributionAnalyzer(
    orig_checkpoint=orig_ckpt,
    tuned_checkpoint=tuned_ckpt,
    comparison_path=comparison_path,
    device=device
)
analyzer.run_analysis_html(max_count=5)
```


---

## Demo Notebook

All deliverables and training details are in the Jupyter notebook:

```
notebooks/demo.ipynb
```

