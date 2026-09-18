# Matrix-chain decomposition study

- horizon T (chain length): 100
- complexity axis n = matrix dimension d: [1, 2, 3, 4, 5, 6, 7, 8]
- base config: `concord/config/matmul_per_role.yaml`
- mock: False   runs: 8

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 1 | 1.00 | 1.000 | 0.520 | 22727 | 0.0982 | 25 |
| 2 | 1 | 0.00 | 0.389 | 1.000 | 609136 | 4.7806 | 492 |
| 3 | 1 | 1.00 | 1.000 | 1.000 | 948111 | 8.3528 | 610 |
| 4 | 1 | 0.00 | 0.357 | 1.000 | 1215922 | 11.3779 | 616 |
| 5 | 1 | 0.00 | 0.224 | 1.000 | 1591849 | 16.5372 | 577 |
| 6 | 1 | 0.00 | n/a | 0.500 | 163828 | 0.7070 | 12 |
| 7 | 1 | 0.00 | 0.000 | 1.000 | 216852 | 0.7999 | 13 |
| 8 | 1 | 0.00 | 0.000 | 1.000 | 268705 | 0.9214 | 13 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 22727 | 0.0982 | OK | 3/3 (1.00) | 0.52 |  |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 609136 | 4.7806 | X | 28/72 (0.39) | 1.00 | Y |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 948111 | 8.3528 | OK | 129/129 (1.00) | 1.00 | Y |
| d4_r1 | 4 | anthropic/claude-opus-4-8 | 1215922 | 11.3779 | X | 45/126 (0.36) | 1.00 | Y |
| d5_r1 | 5 | anthropic/claude-opus-4-8 | 1591849 | 16.5372 | X | 28/125 (0.22) | 1.00 | Y |
| d6_r1 | 6 | anthropic/claude-opus-4-8 | 163828 | 0.7070 | X | 0/0 | 0.50 |  |
| d7_r1 | 7 | anthropic/claude-opus-4-8 | 216852 | 0.7999 | X | 0/1 (0.00) | 1.00 |  |
| d8_r1 | 8 | anthropic/claude-opus-4-8 | 268705 | 0.9214 | X | 0/1 (0.00) | 1.00 |  |
