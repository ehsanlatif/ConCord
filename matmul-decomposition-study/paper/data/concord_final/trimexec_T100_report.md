# Matrix-chain decomposition study

- horizon T (chain length): 100
- complexity axis n = matrix dimension d: [1, 2, 3]
- base config: `concord/config/matmul_per_role_trimexec.yaml`
- mock: False   runs: 9

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 3 | 1.00 | 1.000 | 1.000 | 71108 | 0.3681 | 90 |
| 2 | 3 | 0.67 | 0.904 | 1.000 | 172196 | 1.2907 | 140 |
| 3 | 3 | 1.00 | 1.000 | 1.000 | 233694 | 2.0611 | 139 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 72661 | 0.3802 | OK | 55/55 (1.00) | 1.00 | Y |
| d1_r2 | 1 | anthropic/claude-opus-4-8 | 63103 | 0.3225 | OK | 47/47 (1.00) | 1.00 | Y |
| d1_r3 | 1 | anthropic/claude-opus-4-8 | 77561 | 0.4015 | OK | 59/59 (1.00) | 1.00 | Y |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 169827 | 1.2685 | OK | 81/81 (1.00) | 1.00 | Y |
| d2_r2 | 2 | anthropic/claude-opus-4-8 | 175446 | 1.3190 | OK | 84/84 (1.00) | 1.00 | Y |
| d2_r3 | 2 | anthropic/claude-opus-4-8 | 171316 | 1.2845 | X | 57/80 (0.71) | 1.00 | Y |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 224668 | 2.0006 | OK | 75/75 (1.00) | 1.00 | Y |
| d3_r2 | 3 | anthropic/claude-opus-4-8 | 237152 | 2.0873 | OK | 83/83 (1.00) | 1.00 | Y |
| d3_r3 | 3 | anthropic/claude-opus-4-8 | 239261 | 2.0953 | OK | 85/85 (1.00) | 1.00 | Y |
