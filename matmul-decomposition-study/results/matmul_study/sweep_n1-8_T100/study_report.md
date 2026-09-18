# Matrix-chain decomposition study

- horizon T (chain length): 100
- complexity axis n = matrix dimension d: [1, 2, 3, 4, 5, 6, 7, 8]
- base config: `concord/config/matmul_per_role.yaml`
- mock: False   runs: 8

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 1 | 1.00 | 1.000 | 0.520 | 18830 | 0.0725 | 19 |
| 2 | 1 | 0.00 | 0.371 | 0.700 | 315053 | 2.1987 | 243 |
| 3 | 1 | 0.00 | 0.500 | 0.660 | 358730 | 2.4662 | 239 |
| 4 | 1 | 0.00 | 0.464 | 0.660 | 449882 | 3.3547 | 237 |
| 5 | 1 | 0.00 | 0.500 | 0.640 | 528325 | 4.1045 | 227 |
| 6 | 1 | 0.00 | 0.000 | 1.000 | 174978 | 0.6213 | 14 |
| 7 | 1 | 0.00 | n/a | 0.500 | 214785 | 0.7764 | 13 |
| 8 | 1 | 0.00 | n/a | 0.500 | 255956 | 0.9471 | 12 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 18830 | 0.0725 | OK | 2/2 (1.00) | 0.52 |  |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 315053 | 2.1987 | X | 13/35 (0.37) | 0.70 | Y |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 358730 | 2.4662 | X | 11/22 (0.50) | 0.66 | Y |
| d4_r1 | 4 | anthropic/claude-opus-4-8 | 449882 | 3.3547 | X | 13/28 (0.46) | 0.66 | Y |
| d5_r1 | 5 | anthropic/claude-opus-4-8 | 528325 | 4.1045 | X | 13/26 (0.50) | 0.64 | Y |
| d6_r1 | 6 | anthropic/claude-opus-4-8 | 174978 | 0.6213 | X | 0/1 (0.00) | 1.00 |  |
| d7_r1 | 7 | anthropic/claude-opus-4-8 | 214785 | 0.7764 | X | 0/0 | 0.50 |  |
| d8_r1 | 8 | anthropic/claude-opus-4-8 | 255956 | 0.9471 | X | 0/0 | 0.50 |  |
