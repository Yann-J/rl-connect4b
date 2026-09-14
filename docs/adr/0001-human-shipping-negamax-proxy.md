# Human matches are the shipping bar; negamax is the eval proxy

The PRD treated winrate vs Kaggle negamax as the primary strength metric. We instead ship when a person loses most Search-play Human matches and Policy play does not miss Forced wins or Forced blocks. Winrate vs negamax > 0.5 over 200 Eval matches stays as the automatable proxy so training has a number that is not a handful of Human matches.
