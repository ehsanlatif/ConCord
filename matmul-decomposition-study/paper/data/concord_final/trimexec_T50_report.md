# Matrix-chain decomposition study

- horizon T (chain length): 50
- complexity axis n = matrix dimension d: [1, 2, 3]
- base config: `concord/config/matmul_per_role_trimexec.yaml`
- mock: False   runs: 9

## Solve rate & cost vs complexity n (=d)

| n (=d) | runs | solve rate | mean atom pass | mean fidelity | mean tokens | mean $ | mean calls |
|--|--|--|--|--|--|--|--|
| 1 | 3 | 1.00 | 1.000 | 1.000 | 58695 | 0.3008 | 74 |
| 2 | 3 | 1.00 | 1.000 | 1.000 | 90156 | 0.6510 | 70 |
| 3 | 3 | 0.33 | 0.462 | 1.000 | 121194 | 1.0431 | 71 |

## Per-run results

| run | n | exec model | tokens | $ | overall | atom pass | fidelity | flag |
|--|--|--|--|--|--|--|--|--|
| d1_r1 | 1 | anthropic/claude-opus-4-8 | 56419 | 0.2894 | OK | 41/41 (1.00) | 1.00 | Y |
| d1_r2 | 1 | anthropic/claude-opus-4-8 | 64970 | 0.3348 | OK | 51/51 (1.00) | 1.00 | Y |
| d1_r3 | 1 | anthropic/claude-opus-4-8 | 54697 | 0.2783 | OK | 38/38 (1.00) | 1.00 | Y |
| d2_r1 | 2 | anthropic/claude-opus-4-8 | 92358 | 0.6572 | OK | 42/42 (1.00) | 1.00 | Y |
| d2_r2 | 2 | anthropic/claude-opus-4-8 | 87775 | 0.6377 | OK | 38/38 (1.00) | 1.00 | Y |
| d2_r3 | 2 | anthropic/claude-opus-4-8 | 90336 | 0.6582 | OK | 41/41 (1.00) | 1.00 | Y |
| d3_r1 | 3 | anthropic/claude-opus-4-8 | 128081 | 1.0916 | X | 1/47 (0.02) | 1.00 | Y |
| d3_r2 | 3 | anthropic/claude-opus-4-8 | 118577 | 1.0316 | X | 15/41 (0.37) | 1.00 | Y |
| d3_r3 | 3 | anthropic/claude-opus-4-8 | 116924 | 1.0061 | OK | 41/41 (1.00) | 1.00 | Y |
