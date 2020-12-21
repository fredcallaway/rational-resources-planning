@everywhere using Statistics
pmap(1:10000) do i
    x = 0
    for i in 1:100000000
        X = rand(100, 100)
        x += mean(X * X)
    end
    x
end