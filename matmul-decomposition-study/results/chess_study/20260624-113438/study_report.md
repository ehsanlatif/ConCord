# Chess uci_to_fen parameter study

- problem: `uci_to_fen_easy_6`
- base config: `concord/config/wide_and_deep.yaml`
- max_plies: None  (gold FEN: `4B3/5n2/r3rR2/p7/Ppp1b2p/1PP5/1Kn1N1k1/7n w - - 44 326`)
- mock: False   runs: 6

## Solvability ratio & cost vs each parameter

Each parameter is swept alone (others at baseline); solvability_ratio = solved / repeats.

### `max_plies`

| value | n | solvability ratio | mean subproblem pass-rate | mean tokens | mean $ |
|---|--|--|--|--|--|
| 20 | 1 | 1.00 | 0.250 | 74290 | 0.5824 |
| 40 | 1 | 0.00 | 0.000 | 77450 | 0.5551 |
| 80 | 1 | 0.00 | 0.000 | 59397 | 0.4041 |
| 160 | 1 | 0.00 | 0.000 | 51934 | 0.3611 |
| 320 | 1 | 0.00 | 0.000 | 65693 | 0.4342 |
| 650 | 1 | 0.00 | n/a | 44629 | 0.2041 |

## Per-run results

| run | exec model | N | K_exec | split_depth | tokens | $ | overall | subproblem pass | decomp fidelity |
|---|---|--|--|--|--|--|--|--|--|
| max_plies=20_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 74290 | 0.5824 | ✓ | 1/4 (0.25) | 1.00 |
| max_plies=40_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 77450 | 0.5551 | ✗ | 0/1 (0.00) | 1.00 |
| max_plies=80_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 59397 | 0.4041 | ✗ | 0/2 (0.00) | 1.00 |
| max_plies=160_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 51934 | 0.3611 | ✗ | 0/2 (0.00) | 1.00 |
| max_plies=320_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 65693 | 0.4342 | ✗ | 0/2 (0.00) | 1.00 |
| max_plies=650_r1 | anthropic/claude-sonnet-4-6 | 256 | 2 | 2 | 44629 | 0.2041 | ✗ | 0/0 | n/a |

## Tokens required to reach a success level

| success level | runs reaching it | min tokens | median | max tokens | max $ | cheapest run |
|---|--|--|--|--|--|---|
| overall_correct | 1 | 74290 | 74290 | 74290 | 0.5824 | max_plies=20_r1 |
| subproblems>=1.0 | 0 | — | — | — | — | — |
| subproblems>=0.9 | 0 | — | — | — | — | — |
| subproblems>=0.75 | 0 | — | — | — | — | — |
| subproblems>=0.5 | 0 | — | — | — | — | — |

## Cost / accuracy frontier (best pass-rate at or below a token budget)

| run | total tokens | $ | subproblem pass-rate | overall |
|---|--|--|--|--|
| max_plies=650_r1 | 44629 | 0.2041 | n/a | ✗ |
| max_plies=20_r1 | 74290 | 0.5824 | 0.250 | ✓ |
