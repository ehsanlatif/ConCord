# Chess study regression analysis

Source data: `study_summary.json` and `study_summary.csv`

Runs used: six single-repeat observations for `pipeline.max_split_depth = 1..6`.

## Recommended token-consumption model

Because token use is positive and grows nonlinearly with depth, the most useful fit over the observed range is a quadratic regression on log tokens:

```text
ln(tokens) = 8.308969 + 1.480658 * depth - 0.103118 * depth^2
```

Equivalently:

```text
predicted_tokens(depth) = exp(8.308969 + 1.480658 * depth - 0.103118 * depth^2)
```

Fit diagnostics:

- R^2 on log(tokens): 0.9958
- R^2 on token scale: 0.9726
- RMSE on token scale: 41,924 tokens
- Mean absolute percentage error: 7.0%

Important caveat: use this as an interpolation model for depths 1-6 only. It should not be trusted for much deeper values because depth 6 was `budget_truncated`, and the negative quadratic term will eventually flatten/decrease the curve.

## Solvability model

Solvability is binary here, so the fit is a one-variable logistic regression:

```text
logit(P(solved)) = -4.249097 + 1.214028 * depth
```

Equivalently:

```text
P(solved | depth) = 1 / (1 + exp(4.249097 - 1.214028 * depth))
```

Fit diagnostics:

- Brier score: 0.1468
- McFadden pseudo-R^2: 0.4042
- 50% probability threshold: depth 3.5

Important caveat: this is a monotonic model, but the observed data is not monotonic: depth 3 solved, depth 4 missed, and depths 5-6 solved. With one repeat per depth, the logistic equation should be treated as a smoothed trend, not a reliable probability estimate.

## Fitted values

| depth | observed tokens | predicted tokens | observed solved | predicted solvability |
|---:|---:|---:|:---:|---:|
| 1 | 17,042 | 16,099 | no | 4.6% |
| 2 | 47,600 | 51,938 | no | 13.9% |
| 3 | 136,347 | 136,336 | yes | 35.3% |
| 4 | 278,053 | 291,185 | no | 64.7% |
| 5 | 591,629 | 506,008 | yes | 86.1% |
| 6 | 660,469 | 715,449 | yes | 95.4% |

## Practical readout

- For exact-answer success, depth 3 is the cheapest observed success despite the logistic model assigning it only a 35.3% smoothed probability.
- Depth 4 is the main residual/outlier for solvability: the model predicts it should be more likely to solve than depth 3, but the run failed.
- Depth 5 is the strongest observed quality point, but token use is much higher than depth 3.
- More repeats per depth are needed before treating the solvability equation as anything more than a directional trend.
