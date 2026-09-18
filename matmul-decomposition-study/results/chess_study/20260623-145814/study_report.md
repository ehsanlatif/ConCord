# Chess uci_to_fen parameter study

- problem: `uci_to_fen_easy_6`
- base config: `concord/config/wide_and_deep.yaml`
- max_plies: 20  (gold FEN: `r1bqkb1r/2pp1p1p/n7/p5pP/1p1PpnPR/NP6/P1PQPP2/R1B1KBN1 w Qkq - 1 11`)
- mock: False   runs: 6

## Solvability ratio & cost vs each parameter

Each parameter is swept alone (others at baseline); solvability_ratio = solved / repeats.

### `pipeline.max_split_depth`

| value | n | solvability ratio | mean subproblem pass-rate | mean tokens | mean $ |
|---|--|--|--|--|--|
| 1 | 1 | 0.00 | n/a | 17042 | 0.1324 |
| 2 | 1 | 0.00 | 0.250 | 47600 | 0.3478 |
| 3 | 1 | 1.00 | 0.167 | 136347 | 0.9557 |
| 4 | 1 | 0.00 | 0.125 | 278053 | 1.6042 |
| 5 | 1 | 1.00 | 0.857 | 591629 | 3.4580 |
| 6 | 1 | 1.00 | 0.367 | 660469 | 4.2338 |

## Per-run results

| run | exec model | N | K_exec | split_depth | tokens | $ | overall | subproblem pass | decomp fidelity |
|---|---|--|--|--|--|--|--|--|--|
| max_split_depth=1_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 1 | 17042 | 0.1324 | ✗ | 0/0 | n/a |
| max_split_depth=2_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 2 | 47600 | 0.3478 | ✗ | 1/4 (0.25) | 1.00 |
| max_split_depth=3_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 3 | 136347 | 0.9557 | ✓ | 1/6 (0.17) | 1.00 |
| max_split_depth=4_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 4 | 278053 | 1.6042 | ✗ | 1/8 (0.12) | 0.75 |
| max_split_depth=5_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 5 | 591629 | 3.4580 | ✓ | 6/7 (0.86) | 0.67 |
| max_split_depth=6_r1 | anthropic/claude-sonnet-4-6 | 256 | 1 | 6 | 660469 | 4.2338 | ✓ | 11/30 (0.37) | 0.65 |

## Tokens required to reach a success level

| success level | runs reaching it | min tokens | median | max tokens | max $ | cheapest run |
|---|--|--|--|--|--|---|
| overall_correct | 3 | 136347 | 591629 | 660469 | 4.2338 | max_split_depth=3_r1 |
| subproblems>=1.0 | 0 | — | — | — | — | — |
| subproblems>=0.9 | 0 | — | — | — | — | — |
| subproblems>=0.75 | 1 | 591629 | 591629 | 591629 | 3.458 | max_split_depth=5_r1 |
| subproblems>=0.5 | 1 | 591629 | 591629 | 591629 | 3.458 | max_split_depth=5_r1 |

## Cost / accuracy frontier (best pass-rate at or below a token budget)

| run | total tokens | $ | subproblem pass-rate | overall |
|---|--|--|--|--|
| max_split_depth=1_r1 | 17042 | 0.1324 | n/a | ✗ |
| max_split_depth=2_r1 | 47600 | 0.3478 | 0.250 | ✗ |
| max_split_depth=5_r1 | 591629 | 3.4580 | 0.857 | ✓ |
