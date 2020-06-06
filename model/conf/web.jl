EXPERIMENT = "webofcash-1.4"
# COSTS = 0:0.04:4
COSTS = round.([0; logspace(1e-3, 4, 50); 100]; digits=8)
PRUNE_SAT_THRESHOLDS = -30:5:30
EXPAND_ONLY = true
FIT_BIAS = false