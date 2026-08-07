# QuantAgent quantitative model foundations

These references are design constraints, not decorative citations. A model or
factor change should state which assumption it uses, changes, or empirically
rejects.

1. Sharpe (1964), **Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk**  
   https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x
2. Black & Scholes (1973), **The Pricing of Options and Corporate Liabilities**  
   https://www.cs.princeton.edu/courses/archive/fall09/cos323/papers/black_scholes73.pdf
3. Fama & French (1992), **The Cross-Section of Expected Stock Returns**  
   https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04398.x
4. Carhart (1997), **On Persistence in Mutual Fund Performance**  
   https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1997.tb03808.x
5. Kakushadze (2016), **101 Formulaic Alphas**  
   https://arxiv.org/abs/1601.00991
6. Gu, Kelly & Xiu (2020), **Empirical Asset Pricing via Machine Learning**  
   https://academic.oup.com/rfs/article/33/5/2223/5758276
7. Yang et al. (2020), **Qlib: An AI-oriented Quantitative Investment Platform**  
   https://arxiv.org/abs/2009.11189

## Model rules

- **Risk/return baseline**: market beta/risk premium remains an explicit control;
  alpha claims must be incremental to sensible market/style baselines.
- **Style factors**: size, value and momentum exposures are measured and
  controlled rather than unknowingly re-labelled as proprietary alpha.
- **Formulaic alphas**: operator-style price/volume features may be numerous and
  correlated; selection must account for redundancy and trial multiplicity.
- **Nonlinearity**: distinguish per-factor nonlinear transform from genuine
  cross-factor interaction. A nonlinear label is not accepted unless the
  feature/model class can represent conditional effects.
- **Interaction selection**: pair candidates are selected on training data only,
  after projecting out both parent main effects; the candidate budget is finite,
  deterministic and explicit.
- **Machine learning comparison**: every nonlinear learner is evaluated against
  the same linear baseline, folds, costs and tradability rules. Complexity must
  earn incremental OOS value rather than win by a different protocol.
- **Infrastructure**: Qlib-style reproducibility applies: dataset version,
  features, labels, split, model, prediction and backtest artifacts are linked.

## Data-flow invariants

1. Canonical key is `(trade_date, symbol)` unless the dataset contract explicitly
   declares a different key.
2. `available_at` is the earliest provable decision time. Event/ex-post data may
   be used later than reality when metadata is incomplete, never earlier.
3. Raw prices are the U0 base. Adjustment is explicit and downstream; raw/qfq/hfq
   series are never silently mixed.
4. Units are normalized at ingestion (`volume=shares`, `amount=CNY`) and source
   semantics are persisted.
5. Fit-only transforms/selection use the training segment. Validation/holdout
   labels cannot influence factor selection, pair selection, scaling, tuning or
   champion choice.
6. Generated feature matrices must preserve source row index/order; every
   concatenation/merge must validate keys and duplicate cardinality.
7. Forward-return labels are targets only. Any feature with a timestamp later
   than the prediction decision time is rejected.
8. Missing, entitlement-failed, stale or semantically incompatible data is
   fail-closed; synthetic fallback is never presented as observed market data.
