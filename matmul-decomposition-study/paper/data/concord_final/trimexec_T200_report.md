# Matrix-chain decomposition study

- horizon T (chain length): 200
- complexity axis n = matrix dimension d: [1, 2, 3]
- base config: `concord/config/matmul_per_role_trimexec.yaml`
- mock: False   runs: 9

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 3 | 1.00 | 1.000 | 1.000 | 93662 | 0.4846 | 122 |
| 2 | 3 | 0.67 | 0.877 | 1.000 | 221924 | 1.9428 | 123 |
| 3 | 3 | 0.33 | 0.701 | 1.000 | 312174 | 3.1029 | 121 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 94433 | 0.4892 | OK | 71/71 (1.00) | 1.00 | Y |
| d1_r2 | 1 | anthropic/claude-opus-4-8 | 93648 | 0.4827 | OK | 70/70 (1.00) | 1.00 | Y |
| d1_r3 | 1 | anthropic/claude-opus-4-8 | 92904 | 0.4820 | OK | 69/69 (1.00) | 1.00 | Y |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 225841 | 1.9541 | OK | 74/74 (1.00) | 1.00 | Y |
| d2_r2 | 2 | anthropic/claude-opus-4-8 | 216928 | 1.9102 | X | 43/68 (0.63) | 1.00 | Y |
| d2_r3 | 2 | anthropic/claude-opus-4-8 | 223004 | 1.9640 | OK | 70/70 (1.00) | 1.00 | Y |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 305341 | 3.0480 | X | 14/67 (0.21) | 1.00 | Y |
| d3_r2 | 3 | anthropic/claude-opus-4-8 | 308927 | 3.0838 | OK | 68/68 (1.00) | 1.00 | Y |
| d3_r3 | 3 | anthropic/claude-opus-4-8 | 322253 | 3.1769 | X | 67/75 (0.89) | 1.00 | Y |
