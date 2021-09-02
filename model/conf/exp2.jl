EXPERIMENT = "exp2"
EXPAND_ONLY = true

MAX_VALUE = 50.
MAX_DEPTH = 5.

QUOTE_MODELS = quote 
    [
        #OptimalPlus{:Default},
        #MetaGreedy{:Default},
        #Heuristic{:Random},

        all_heuristic_models()...
    ] 
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,
        
        Heuristic{:Best_Full},
        Heuristic{:Depth_Full},
        Heuristic{:Breadth_Full},
        
        # Heuristic{:Best_Satisfice_BestNext_Prune},
        # Heuristic{:Depth_Satisfice_BestNext_Prune},
        # Heuristic{:Breadth_Satisfice_BestNext_Prune},
        
        # Heuristic{:Best_Satisfice_BestNext},
        # Heuristic{:Depth_Satisfice_BestNext},
        # Heuristic{:Breadth_Satisfice_BestNext},
    ]
end