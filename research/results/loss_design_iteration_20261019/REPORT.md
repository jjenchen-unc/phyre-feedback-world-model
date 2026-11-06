# Loss design iterations — results

Same frozen Q0, classifier top-1024 candidate pool, Q0 first attempt, 9-attempt budget. All new updaters use raw residual during training. Each protocol has 100 evaluation scenes.

## cross

| Method, seed0 | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| l1 | 3 | 7 | 10 | 12 |
| mse | 3 | 7 | 8 | 8 |
| smooth | 3 | 7 | 10 | 12 |
| l1_centered | 3 | 6 | 10 | 15 |
| l1_zero | 3 | 7 | 7 | 9 |
| mse_centered | 3 | 7 | 9 | 10 |
| mse_zero | 3 | 7 | 9 | 10 |
| smooth_centered | 3 | 9 | 12 | 15 |
| smooth_zero | 3 | 10 | 12 | 14 |

No new raw-input loss met the predeclared S@9 gain without S@5 regression; no second training round launched. Use fixed-query and input controls to guide next work.


Fixed-query diagnostics compare identical first support, observation and eight query actions. Negative error changes mean lower error, but do not establish better closed-loop success.

```json
{
  "mse": {
    "train": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": 0.0066957068629562855,
        "mse": 0.0013025703374296427,
        "geometry": 0.09646221809089184,
        "mean_bias_mse": 6.361761188600212e-05,
        "centered_mse": 0.0012389500625431538,
        "update_norm": 1.3870195746421814
      },
      "next_successes": 1,
      "l1_next_successes": 0,
      "top8_successes": 2,
      "l1_top8_successes": 2
    },
    "evaluation": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": 0.0037176823243498802,
        "mse": 0.0014520585536956787,
        "geometry": 0.10243052244186401,
        "mean_bias_mse": 0.00016013826825655997,
        "centered_mse": 0.0012919194996356964,
        "update_norm": 1.5391070544719696
      },
      "next_successes": 1,
      "l1_next_successes": 1,
      "top8_successes": 1,
      "l1_top8_successes": 2
    }
  },
  "smooth": {
    "train": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": -0.002923495601862669,
        "mse": -0.004471883410587907,
        "geometry": 0.039591921493411064,
        "mean_bias_mse": -0.0001166028123407159,
        "centered_mse": -0.004355282289907336,
        "update_norm": -0.3976903557777405
      },
      "next_successes": 2,
      "l1_next_successes": 0,
      "top8_successes": 2,
      "l1_top8_successes": 2
    },
    "evaluation": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": -0.0003123730421066284,
        "mse": -0.0007080030627548695,
        "geometry": -0.025752224028110504,
        "mean_bias_mse": -3.621136420406401e-05,
        "centered_mse": -0.0006717960350215435,
        "update_norm": -0.24931463599205017
      },
      "next_successes": 0,
      "l1_next_successes": 1,
      "top8_successes": 1,
      "l1_top8_successes": 2
    }
  }
}
```

## within

| Method, seed0 | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| l1 | 29 | 45 | 50 | 56 |
| mse | 29 | 42 | 50 | 56 |
| smooth | 29 | 46 | 51 | 61 |
| l1_centered | 29 | 46 | 51 | 57 |
| l1_zero | 29 | 41 | 49 | 55 |
| mse_centered | 29 | 42 | 52 | 56 |
| mse_zero | 29 | 39 | 49 | 54 |
| smooth_centered | 29 | 48 | 54 | 61 |
| smooth_zero | 29 | 43 | 53 | 57 |

Round1 improvement did not replicate as a positive S@9 difference; no robust winner established.

| Method, seed1 | S@1 | S@3 | S@5 | S@9 |
|---|---:|---:|---:|---:|
| l1 | 29 | 45 | 49 | 58 |
| smooth | 29 | 40 | 47 | 50 |

Fixed-query diagnostics compare identical first support, observation and eight query actions. Negative error changes mean lower error, but do not establish better closed-loop success.

```json
{
  "mse": {
    "train": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": 0.0005638301372528076,
        "mse": -0.0009720991365611553,
        "geometry": 0.0325850285589695,
        "mean_bias_mse": -8.932649507187307e-05,
        "centered_mse": -0.0008827731944620609,
        "update_norm": 0.5905789732933044
      },
      "next_successes": 2,
      "l1_next_successes": 3,
      "top8_successes": 3,
      "l1_top8_successes": 7
    },
    "evaluation": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": 0.0017603300511837006,
        "mse": 0.0016917451284825802,
        "geometry": 0.04694828577339649,
        "mean_bias_mse": -7.789037772454321e-05,
        "centered_mse": 0.0017696365248411894,
        "update_norm": 0.9502891898155212
      },
      "next_successes": 1,
      "l1_next_successes": 0,
      "top8_successes": 3,
      "l1_top8_successes": 1
    }
  },
  "smooth": {
    "train": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": -0.004222075454890728,
        "mse": -0.005074087996035814,
        "geometry": -0.03953286446630955,
        "mean_bias_mse": -0.0003856041294056922,
        "centered_mse": -0.004688484128564596,
        "update_norm": 1.148980438709259
      },
      "next_successes": 1,
      "l1_next_successes": 3,
      "top8_successes": 5,
      "l1_top8_successes": 7
    },
    "evaluation": {
      "n": 8,
      "mean_changes_vs_l1": {
        "l1": -0.0004191845655441284,
        "mse": -0.0012557876762002707,
        "geometry": -0.01704806461930275,
        "mean_bias_mse": -0.0002920566766988486,
        "centered_mse": -0.0009637321345508099,
        "update_norm": 0.9987209439277649
      },
      "next_successes": 1,
      "l1_next_successes": 0,
      "top8_successes": 1,
      "l1_top8_successes": 1
    }
  }
}
```

## Interpretation limits

Loss coefficients were fixed using training-only final-memory-state gradient norms. This does not guarantee equal full-parameter gradients throughout training. Seed1 changes updater initialization and episode sampling, not the pretrained Q0 or fold. Centered/zero evaluation inputs are interventions. Zero-input performance is not proof of semantic feedback use or non-use. No training-loss-only winner is selected.
