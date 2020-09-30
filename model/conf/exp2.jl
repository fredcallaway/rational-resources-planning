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

        Heuristic{:Best_Full},
        Heuristic{:Best_Full_NoPrune},
        Heuristic{:Breadth_Full},
        Heuristic{:Breadth_Full_NoPrune},
        Heuristic{:Depth_Full},
        Heuristic{:Depth_Full_NoPrune},
    ] 
end

        # OptimalPlus{:Default},
        # MetaGreedy{:Default},
        # Heuristic{:Random},
        # Heuristic{:BestFirst},
        # Heuristic{:DepthFirst},
        # Heuristic{:BreadthFirst},
        # Heuristic{:BestFirstNoPrune},
        # Heuristic{:DepthFirstNoPrune},
        # Heuristic{:BreadthFirstNoPrune},
