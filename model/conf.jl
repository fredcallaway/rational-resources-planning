using Distributed
if myid() == 1  # only run on the master process
    if isempty(ARGS)
        if isinteractive()
            print("Experiment: ")
            x = readline()
            push!(ARGS, "exp" * replace(x, "exp" => ""))
        end
        error("Must pass configuration name as first argument!")
    end
    conf = ARGS[1]
    @everywhere begin
        # defaults
        COSTS = [0:0.05:4; 100]
        FOLDS = 5
        CV_METHOD = :random  # :stratified
        OPT_METHOD = :bfgs

        conf = $conf
        include("conf/$conf.jl")
        base_path = "tmp/$EXPERIMENT"
        results_path = "results/$EXPERIMENT"
    end
    mkpath(base_path)
    mkpath(results_path)
end