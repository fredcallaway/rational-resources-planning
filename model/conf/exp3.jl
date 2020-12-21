EXPERIMENT = "exp3"
EXPAND_ONLY = false

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
