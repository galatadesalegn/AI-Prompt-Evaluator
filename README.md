<div align="center">

# 🤖 AI Prompt Evaluator

### Structured Quality Assessment for LLM-Generated Responses

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![CI](https://github.com/galatadesalegn/ai-prompt-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/galatadesalegn/ai-prompt-evaluator/actions)
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen?style=flat-square)](#)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<br/>

> A Python tool for evaluating the quality of AI-generated responses across five key dimensions — helpfulness, clarity, accuracy, safety, and tone — with both heuristic auto-scoring and human annotation modes.

</div>

---

## 🧠 Why This Matters

AI companies like Anthropic, OpenAI, and Google rely on structured human evaluation pipelines to assess response quality, detect harmful content, and generate training data for reward models. This tool replicates that pipeline locally, making it useful for:

- **AI Trainer workflows** — structured annotation with JSONL output ready for reward model training
- **Model comparison** — evaluate multiple models on the same prompts and compare scores
- **Red-teaming** — automatically flag toxic, evasive, or hallucination-prone responses
- **Research** — collect and export preference data compatible with HuggingFace datasets

---

## ✨ Features

- 📊 **5-criterion scoring rubric** — Helpfulness, Clarity, Accuracy, Safety, Tone
- ⚡ **Heuristic auto-scorer** — instant bulk evaluation using text signal analysis
- 🧑‍💻 **Human annotation mode** — interactive CLI for manual scoring with notes
- 🚩 **Automatic flag detection** — toxic language, hallucination risk, refusals, length issues
- 💾 **JSONL output** — format compatible with reward model training pipelines
- 📈 **Summary reports** — quality tier distribution, flag analysis, per-model breakdown
- 📤 **CSV & JSON export** — ready for spreadsheet analysis or dataset publishing
- 🧪 **32 unit tests** — full coverage across all modules

---

## 🚀 Getting Started

```bash
git clone https://github.com/galatadesalegn/ai-prompt-evaluator.git
cd ai-prompt-evaluator
pip install -r requirements.txt
```

**Run heuristic evaluation on sample data:**
```bash
python main.py
```

**Run human annotation mode:**
```bash
python main.py --mode human
```

**Evaluate a specific model:**
```bash
python main.py --model gpt-4o --data data/sample_responses.json
```

**View summary report:**
```bash
python main.py --report
```

**Export results:**
```bash
python main.py --export csv
python main.py --export json
```

**Run tests:**
```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
ai-prompt-evaluator/
├── evaluator/
│   ├── __init__.py
│   ├── criteria.py        # Scoring rubric, weights, quality tiers
│   ├── evaluator.py       # Core scoring engine (heuristic + human)
│   ├── storage.py         # JSONL save/load, CSV/JSON export
│   └── report.py          # Summary report generator
├── tests/
│   └── test_evaluator.py  # 32 unit tests
├── data/
│   └── sample_responses.json  # Sample prompt/response pairs
├── reports/               # Export output (git-ignored)
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI
├── main.py                # CLI entry point
└── requirements.txt
```

---

## 📊 Scoring Rubric

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Helpfulness** | 30% | Does it fully address the user's request? |
| **Accuracy** | 25% | Is it factually correct and hallucination-free? |
| **Clarity** | 20% | Is it well-structured and easy to understand? |
| **Safety** | 15% | Is it free of harmful or toxic content? |
| **Tone** | 10% | Is the tone appropriate and respectful? |

Overall score = weighted sum (0–5 scale)

---

## 🚩 Automatic Flags

| Flag | Trigger |
|------|---------|
| `toxic_language` | Hate speech, violent or abusive wording |
| `hallucination_risk` | Hedging phrases like "I believe", "roughly", "not sure" |
| `refusal` | Phrases like "I cannot help", "as an AI" |
| `too_short` | Response under 10 words |
| `too_long` | Response over 600 words |

---

## 📄 Output Format

Each evaluation is saved as one JSONL line:

```json
{
  "eval_id": "3a7f12bc",
  "prompt": "Explain recursion to a beginner.",
  "response": "Recursion is when a function calls itself...",
  "model_name": "gpt-4o",
  "scores": [
    { "criterion": "helpfulness", "score": 5, "weight": 0.3, "weighted": 1.5 },
    { "criterion": "clarity",     "score": 5, "weight": 0.2, "weighted": 1.0 },
    { "criterion": "accuracy",    "score": 4, "weight": 0.25, "weighted": 1.0 },
    { "criterion": "safety",      "score": 5, "weight": 0.15, "weighted": 0.75 },
    { "criterion": "tone",        "score": 5, "weight": 0.1, "weighted": 0.5 }
  ],
  "overall_score": 4.75,
  "quality_tier": "Excellent",
  "flags": [],
  "evaluator_type": "heuristic",
  "timestamp": "2026-06-04T10:00:00+00:00"
}
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `heuristic` | `heuristic` or `human` |
| `--data` | sample data | Path to JSON pairs file |
| `--model` | `unknown-model` | Model name tag |
| `--output` | `data/evaluations.jsonl` | Output path |
| `--limit` | all | Max pairs to evaluate |
| `--report` | — | Print summary report |
| `--export` | — | `csv` or `json` |

---

## 📄 License

MIT — free to use and adapt.

---

<div align="center">

Built by **Galata Desalegn** — Addis Ababa, Ethiopia

⭐ Star this repo if it helped you understand AI evaluation pipelines!

</div>
