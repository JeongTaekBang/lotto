# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-model ensemble system for Korean Lotto (6/45) number prediction. Combines 7 ML/DL models with rule-based filters and MMR diversity selection. Written in Python 3.10+ with PyTorch, XGBoost, and scikit-learn.

**This is a learning/analysis project, not a gambling tool.** Random expected hits = 0.80 per draw.

## Commands

```bash
# Environment
conda activate lotto
pip install -r requirements.txt

# Train
python main_new.py train --model=all           # All models (extended features, XGBoost excluded)
python main_new.py train --model=gru            # Single model
python main_new.py train --model=xgboost        # XGBoost (uses basic features automatically)

# Predict
python main_new.py predict --model=cnn_grid --mode=balanced
python main_new.py predict --ensemble --auto-weight --mode=balanced
python main_new.py predict --ensemble --auto-weight --filter

# Evaluate
python main_new.py compare --rounds=100         # Compare all models
python main_new.py evaluate --model=gru --rounds=100

# Utilities
python main_new.py crawl                        # Report DB/API round gap + git pull the blog repo
python main_new.py crawl --no-pull              # Status check only
python main_new.py analyze                      # Statistics report
python main_new.py list                         # List available models
python analysis/visualize_grid.py               # CNN Grid visualization (4 PNGs)

# Tests
pytest                                          # All tests
pytest tests/test_cnn_grid.py                   # Single test file
pytest tests/test_models.py -k "test_gru"       # Single test

# Interactive menu
./lotto.sh     # macOS
lotto.bat      # Windows
```

## Architecture

### Core Abstractions (`core/`)
- `BaseModel` (ABC): All models implement `train()`, `predict_proba()`, `save()`, `load()`, `requires_sequence`
- `BaseDataSource` (ABC): Data loading interface
- `BaseFilter` (ABC): Filter interface with `apply()` (pass/fail) and `score()` (0-1 continuous)
- `BaseEnsemble` (ABC): Ensemble strategy interface
- Key types: `Prediction`, `ProbabilityDistribution` (45-element array), `LottoRecord`, `EvaluationResult`, `TrainingHistory`

### Models (`models/`)
7 models registered via `ModelFactory` (Factory Pattern with class registry):
- **Sequence models** (require 3D input `[batch, seq_len, features]`): `transformer`, `lstm`, `gru`, `cnn_grid`
- **Flat models** (require 2D input `[batch, features]`): `xgboost`, `random_forest`, `markov`
- CNN Grid is unique: reshapes numbers into 7x7 spatial grid, uses multi-branch CNN (3x3, 5x5, 1x1 filters)
- Saved as `.pt` (PyTorch) or `.pkl` (sklearn/xgboost) in `saved_models/`

### Feature Modes
| Mode | Dims | Use case |
|------|------|----------|
| `basic` | 45 | XGBoost standalone (multi-hot only) |
| `extended` | 74 | All other models, ensemble (multi-hot + 29 statistical features) |
| `xgboost` | 66 | XGBoost with deduplicated extended features |

**Critical rule**: XGBoost uses `basic` features; all other models use `extended`. Ensemble always uses `extended` and excludes XGBoost. This is hardcoded in `main_new.py` and `lotto.bat`/`lotto.sh`.

### Ensemble System (`ensemble/`)
- `EnsembleManager`: Orchestrates models, strategies, filters, auto-weight
- Strategies (Strategy Pattern): `WeightedAverageEnsemble`, `VotingEnsemble`, `StackingEnsemble`
- Auto-weight: backtest-based performance weighting (softmax/linear/rank methods)

### Selection & Filters
- `CandidateSelector` (`core/selector.py`): Generates candidates -> scores (log-prob + filter) -> MMR diversity reranking
- 3 presets: `safe` (high diversity), `balanced` (default), `aggressive` (score-focused)
- Filters (Composite Pattern): `PatternFilter`, `StatisticalFilter`, `FrequencyFilter` -> `CompositeFilter`

### Data Pipeline
- `SQLiteDataSource` (`datasources/sqlite_source.py`): **Read-only** reader for the bjt-blog repo's SQLite file (`draws` table: `round`, `draw_date`, `n1`-`n6`, `bonus`, `rank1_winners`, `rank1_amount`, `total_sales`)
  - Path resolution: constructor arg -> `LOTTO_DB_PATH` env var -> default `../bjt-blog/data/lotto.db` (relative to repo root)
  - Opens the DB with a `mode=ro` URI, so the blog's file is never written
  - When the dhlottery API (`selectPstLt645Info.do`) is ahead of the DB, missing rounds are supplemented **in memory only**; disable with `--no-fetch` / `fetch_missing=False`
- `FeatureExtractor` (`core/feature_extractor.py`): Converts records to feature vectors
- `UnifiedTrainer` (`training/trainer.py`): Handles sequence preparation, train/val split, model creation
- `main_new.py crawl`: Reports the DB/API round gap and runs `git pull --ff-only` in the blog repo (`--no-pull` to skip). It never writes to the DB itself.
- No DB server or `.env` credentials required; `.env` is optional and only holds `LOTTO_DB_PATH`

### Adding a New Model
1. Create `models/my_model.py`, subclass `BaseModel`
2. Implement `train()`, `predict_proba()`, `save()`, `load()`, `requires_sequence`
3. Register in `models/factory.py`: `ModelFactory.register('my_model', MyModel)`

## Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

<!-- BEGIN:forrest-project-protocol -->
@AGENTS.md
<!-- END:forrest-project-protocol -->
