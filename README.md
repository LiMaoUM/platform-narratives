# Same Event, Different Truths

Code for the ICWSM paper **"Same Event, Different Truths: How Users Narrate Political Events Across Emerging Social Media Platforms"**.

We study how factual political events (the Trump criminal trial, the Hunter Biden trial, and the first Biden-Trump debate, June 2024) are narrated across three emerging platforms: **Truth Social, Bluesky, and Mastodon**. The pipeline matches discussions of the same event across platforms, classifies narrative frames and reply behavior with an LLM, and quantifies cross-platform narrative divergence.

## Pipeline overview

1. **Data loading and matching** — load platform post/reply dumps, clean text, filter to English, and build cross-platform similarity graphs (sentence embeddings) to match discussions of the same event across platforms.
2. **Anchor selection** — identify significant posts with FastLexRank and extract their conversation trees.
3. **Narrative frame classification** — soft-label each post over a ten-frame taxonomy (Persecution, Corruption, Accountability, Irony/Detachment, Heroism, Civic Critique, Moral Decay, Media Manipulation, Strategic Pragmatism, Cultural Identity) using **Gemma-3-27B-it served by vLLM**.
4. **Target and attitude attribution** — attribute each post's stance toward Trump / Biden / other (positive, neutral, negative).
5. **Reply classification** — classify replies as Reinforce / Challenge / Shift relative to the root post's narrative.
6. **Divergence and engagement analysis** — compute Narrative Divergence Index (NDI) variants, reinforce rates, and engagement GLMs.
7. **Validation** — human coding of samples and Krippendorff's alpha / accuracy against LLM labels.

## Repository layout

```
├── unified_analysis.py     # Main CLI: runs the pipeline steps end to end
├── config.yaml             # Pipeline configuration (data paths, models, thresholds)
├── src/                    # Modular pipeline components
│   ├── cross_platform_analyzer.py       # Cross-platform matching + orchestration
│   ├── reply_analyzer.py                # Reply extraction and classification
│   ├── narrative_classification.py      # LLM frame soft-labeling
│   ├── narrative_divergence_analyzer.py # NDI metrics
│   ├── target_attitude_analysis.py      # Stance toward Trump/Biden/other
│   ├── event_detection.py               # Event tagging
│   ├── vllm_wrapper.py                  # Gemma-3-27B-it via vLLM
│   └── similarity_graph.py, lexrank.py, ranking.py, graph_analysis.py,
│       narrative_trees.py, reply_chain_analysis.py, text_processing.py,
│       config_manager.py, utils.py
├── notebooks/              # Statistical analysis behind the paper's results
│   ├── matching.ipynb              # Data preparation and cross-platform matching
│   ├── analysis_workflow.ipynb     # Pipeline walkthrough + NDI computation
│   ├── analysis.ipynb              # GLMs, validation (Krippendorff's alpha), figures
│   └── *.csv                       # Derived coefficient/validation tables (no post content)
├── legacy/                 # Scripts as originally run for the paper (provenance)
└── tests/                  # Batch-processing and structure tests
```

`legacy/` contains the original monolithic scripts that produced the paper's runs, kept verbatim for provenance. `src/` + `unified_analysis.py` is the maintained, refactored version of the same pipeline.

## Installation

Requires Python ≥ 3.10 and, for LLM classification, GPUs able to serve Gemma-3-27B-it (the paper used vLLM with `tensor_parallel_size=2` on A100-80GB GPUs).

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -r requirements.txt
```

## Usage

```bash
# Run the full pipeline
uv run unified_analysis.py --steps all

# Run specific steps
uv run unified_analysis.py --steps setup load_data cross_platform_matching

# Resume from a step
uv run unified_analysis.py --steps all --resume-from narrative_analysis

# Custom config
uv run unified_analysis.py --config my_config.yaml --steps all
```

Configure data paths, the embedding model, the vLLM model, and classification settings in `config.yaml`.

## Data availability

Raw platform data is not redistributed, in line with platform terms of service. The repository ships only derived statistical tables (model coefficients, agreement metrics) that contain no post content. To reproduce the pipeline, collect post/reply data from Truth Social, Bluesky, and Mastodon (keyword search plus recursive context retrieval, as described in the paper) and point `config.yaml` at your local copies under `data/`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li_sameevent,
  title     = {Same Event, Different Truths: How Users Narrate Political Events Across Emerging Social Media Platforms},
  author    = {Li, Mao and et al.},  % TODO: final author list
  booktitle = {Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)},
  year      = {2027}                 % TODO: confirm volume year
}
```

## License

MIT — see [LICENSE](LICENSE).
