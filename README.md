# ConCord

**Adaptive Tree-search with Unanimous-Confidence Decomposition** — a research
prototype for LLM problem-solving by Monte-Carlo-Tree-Search over decompositions
(Split → Solve → Combine → Verify → Synthesize).

This repository consolidates every development branch of the project into a
**single `main` branch**, with **one folder per original branch** so all content
lives together while remaining separable:

| Folder | What it contains |
|---|---|
| [`matmul-decomposition-study/`](matmul-decomposition-study/) | **The current / paper branch.** The full ConCord package plus the matrix-chain long-horizon-execution study: method, benchmark dataset, reproduction scripts, results, figures, and the paper companion (`paper/`). Start here. |
| [`main/`](main/) | The base ConCord prototype (package, spec, implementation plan) prior to the matmul study. |

## Where to start

- **Method & contributions:** `matmul-decomposition-study/paper/02_method_and_contributions.md`
- **Results & figures:** `matmul-decomposition-study/paper/04_main_results.md`, `matmul-decomposition-study/paper/figures/`
- **Reproduction scripts:** `matmul-decomposition-study/scripts/`
- **Benchmark dataset:** `matmul-decomposition-study/data/matmul/` (static, golden-labeled; sha256-verified in `config.json`)
- **Core package:** `matmul-decomposition-study/concord/`

## Reproduce

```bash
cd matmul-decomposition-study
pip install -r requirements.txt
# unit tests (no API calls)
python -m pytest concord/tests/ -q
# live experiments require an ANTHROPIC_API_KEY (and OPENAI_API_KEY for GPT baselines)
```

See each folder's own `README.md` for details.

## License

MIT — see [`LICENSE`](matmul-decomposition-study/LICENSE).
