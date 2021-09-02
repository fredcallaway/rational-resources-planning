EXPERIMENT = "exp3"
EXPAND_ONLY = false

MAX_VALUE = 30.
MAX_DEPTH = 3

MDP_IDS = [
    "6oCv6ld5V89",
    "IXcheAhOWH7",
    "95YVmaymRm9",
]

QUOTE_MODELS = quote 
    [
        OptimalPlus{:Default},
        MetaGreedy{:Default},
        Heuristic{:Random},

        OptimalPlus{:Expand},
        MetaGreedy{:Expand},
        Heuristic{:Expand},

        all_heuristic_models()...
    ]
end

QUOTE_PARETO_MODELS = quote
    [
        RandomSelection,
        MetaGreedy,

        Heuristic{:Best_Satisfice_BestNext_Expand},
        Heuristic{:Depth_Satisfice_BestNext_Expand},
        Heuristic{:Breadth_Satisfice_BestNext_Expand},
        Heuristic{:Best_Satisfice_BestNext},
        Heuristic{:Depth_Satisfice_BestNext},
        Heuristic{:Breadth_Satisfice_BestNext},
    ]
end