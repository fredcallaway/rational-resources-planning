using Distributed
if myid() == 1  # only run on the master process
    conf = ARGS[1]
    @everywhere begin
        include("utils.jl")
        conf = $conf
        include("conf/$conf.jl")
        base_path = "tmp/$EXPERIMENT"
        results_path = "results/$EXPERIMENT"
    end
    mkpath(base_path)
    mkpath(results_path)
end