# Matrix-chain decomposition study

- horizon T (chain length): 12
- complexity axis n = matrix dimension d: [1, 2, 3]
- base config: `concord/config/matmul_per_role_trimexec.yaml`
- mock: False   runs: 9

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 3 | 1.00 | 1.000 | 1.000 | 22310 | 0.1072 | 26 |
| 2 | 3 | 1.00 | 1.000 | 1.000 | 30032 | 0.1880 | 27 |
| 3 | 3 | 1.00 | 1.000 | 1.000 | 34423 | 0.2719 | 21 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 21444 | 0.1039 | OK | 11/11 (1.00) | 1.00 |  |
| d1_r2 | 1 | anthropic/claude-opus-4-8 | 23838 | 0.1132 | OK | 17/17 (1.00) | 1.00 |  |
| d1_r3 | 1 | anthropic/claude-opus-4-8 | 21649 | 0.1045 | OK | 11/11 (1.00) | 1.00 |  |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 30205 | 0.1914 | OK | 13/13 (1.00) | 1.00 |  |
| d2_r2 | 2 | anthropic/claude-opus-4-8 | 29119 | 0.1785 | OK | 13/13 (1.00) | 1.00 |  |
| d2_r3 | 2 | anthropic/claude-opus-4-8 | 30772 | 0.1941 | OK | 15/15 (1.00) | 1.00 |  |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 37602 | 0.2873 | OK | 11/11 (1.00) | 1.00 |  |
| d3_r2 | 3 | anthropic/claude-opus-4-8 | 32949 | 0.2643 | OK | 7/7 (1.00) | 1.00 |  |
| d3_r3 | 3 | anthropic/claude-opus-4-8 | 32718 | 0.2640 | OK | 7/7 (1.00) | 1.00 |  |
