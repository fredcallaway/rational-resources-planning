using Distributed
if myid() == 1  # only run on the master process
    conf = ARGS[1]
    @everywhere begin
        conf = $conf
        include("conf/$conf.jl")
        base_path = "tmp/$EXPERIMENT$MODEL_VERSION"
    end
    mkpath(base_path)
end