EXPERIMENT = "exp2"
COSTS = [0:0.05:4; 100]
MAX_THETA = 60
EXPAND_ONLY = true
FIT_BIAS = false

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        Heuristic{:Best},
        Heuristic{:Best_Satisfice},
        Heuristic{:Best_BestNext},
        Heuristic{:Best_DepthLimit},
        Heuristic{:Best_Prune},
        Heuristic{:Best_Full},
        Heuristic{:Best_Full_NoSatisfice},
        Heuristic{:Best_Full_NoBestNext},
        Heuristic{:Best_Full_NoDepthLimit},
        Heuristic{:Best_Full_NoPrune},
        Heuristic{:Best_Satisfice_BestNext},

        Heuristic{:Breadth},
        Heuristic{:Breadth_Satisfice},
        Heuristic{:Breadth_BestNext},
        Heuristic{:Breadth_DepthLimit},
        Heuristic{:Breadth_Prune},
        Heuristic{:Breadth_Full},
        Heuristic{:Breadth_Full_NoSatisfice},
        Heuristic{:Breadth_Full_NoBestNext},
        Heuristic{:Breadth_Full_NoDepthLimit},
        Heuristic{:Breadth_Full_NoPrune},
        Heuristic{:Breadth_Satisfice_BestNext},

        Heuristic{:Depth},
        Heuristic{:Depth_Satisfice},
        Heuristic{:Depth_BestNext},
        Heuristic{:Depth_DepthLimit},
        Heuristic{:Depth_Prune},
        Heuristic{:Depth_Full},
        Heuristic{:Depth_Full_NoSatisfice},
        Heuristic{:Depth_Full_NoBestNext},
        Heuristic{:Depth_Full_NoDepthLimit},
        Heuristic{:Depth_Full_NoPrune},
        Heuristic{:Depth_Satisfice_BestNext},
    ] 
end