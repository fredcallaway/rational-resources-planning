EXPERIMENT = "exp2"
EXPAND_ONLY = true
FIT_BIAS = false

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        all_heuristic_models()...
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        
        Heuristic{:Best_NoDepthLimit},
        Heuristic{:Best_NoDepthLimit_NoPrune},

        Heuristic{:Depth_NoDepthLimit},
        Heuristic{:Breadth_NoDepthLimit}
    ]
end