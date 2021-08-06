Space = OrderedDict{Symbol,Any}

function bounds(space::Space)
    x = (Float64[], Float64[], Float64[], Float64[])
    for spec in values(space)
        if length(spec) == 4
            for i in 1:4
                push!(x[i], spec[i])
            end
        end
    end
    x
end

function combinations(space::Space)
    specs = filter(collect(values(space))) do spec
        spec isa Vector
    end
    Iterators.product(specs...)
end
