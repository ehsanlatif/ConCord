# Matrix-chain decomposition study

- horizon T (chain length): 25
- complexity axis n = matrix dimension d: [1, 2, 3]
- base config: `concord/config/matmul_per_role_trimexec.yaml`
- mock: False   runs: 9

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 3 | 1.00 | 1.000 | 1.000 | 43277 | 0.2130 | 55 |
| 2 | 3 | 1.00 | 1.000 | 1.000 | 55980 | 0.3706 | 50 |
| 3 | 3 | 1.00 | 1.000 | 1.000 | 83240 | 0.6488 | 57 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 48718 | 0.2362 | OK | 34/34 (1.00) | 1.00 |  |
| d1_r2 | 1 | anthropic/claude-opus-4-8 | 46303 | 0.2259 | OK | 29/29 (1.00) | 1.00 |  |
| d1_r3 | 1 | anthropic/claude-opus-4-8 | 34811 | 0.1768 | OK | 25/25 (1.00) | 1.00 |  |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 62609 | 0.3974 | OK | 29/29 (1.00) | 1.00 |  |
| d2_r2 | 2 | anthropic/claude-opus-4-8 | 49840 | 0.3439 | OK | 23/23 (1.00) | 1.00 |  |
| d2_r3 | 2 | anthropic/claude-opus-4-8 | 55492 | 0.3703 | OK | 26/26 (1.00) | 1.00 |  |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 86738 | 0.6888 | OK | 30/30 (1.00) | 1.00 |  |
| d3_r2 | 3 | anthropic/claude-opus-4-8 | 84711 | 0.6515 | OK | 30/30 (1.00) | 1.00 |  |
| d3_r3 | 3 | anthropic/claude-opus-4-8 | 78271 | 0.6059 | OK | 25/25 (1.00) | 1.00 |  |
