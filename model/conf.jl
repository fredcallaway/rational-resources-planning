using Distributed
if myid() == 1  # only run on the master process
    if isempty(ARGS)
        error("Must pass configuration name as first argument!")
    end
    conf = ARGS[1]
    @everywhere begin

        # defaults
        CV_METHOD = :random  # :stratified
        OPT_METHOD = :bfgs

        include("utils.jl")
        conf = $conf
        include("conf/$conf.jl")
        base_path = "tmp/$EXPERIMENT"
        results_path = "results/$EXPERIMENT"
    end
    mkpath(base_path)
    mkpath(results_path)
end