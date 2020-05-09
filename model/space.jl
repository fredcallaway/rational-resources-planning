Space = OrderedDict{Symbol,Any}

function bounds(space::Space)
    lo, hi = Float64[], Float64[]
    for spec in values(space)
        if spec isa Tuple
            push!(lo, spec[1]); push!(hi, spec[2])
        end
    end
    lo, hi
end

function combinations(space::Space)
    specs = filter(collect(values(space))) do spec
        spec isa Vector
    end
    Iterators.product(specs...)
end
