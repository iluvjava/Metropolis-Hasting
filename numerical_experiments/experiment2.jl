### Sampling circular domain function in a bounded box via MCH with a base chain 
### of Uniform Distributions on the Box. 

include("../src/core.jl")
using Plots, Distributions, LinearAlgebra


"""
sample uniformally from a box in 2D, this is used as a base chain for 2D MCMC
method. 

------
### Positional Arguments
- `lower_left::Tuple{Reals, Reals}`: A tuple representing the right bottom 
coordinate of the rectagular region. 
- `upper_right::Tuple{Reals, Reals}`: A tuple represetning the top right coordinate 
of the bounding rectangle. 

------
### Returns: 
`[x,y]`: A vector representing the coordinate of the randomly chosen point. 
"""
function unif_sampler_2d(lower_left::Tuple{Real, Real}, upper_right::Tuple{Real, Real})
    unifx = Uniform(lower_left[1], upper_right[1])
    unify = Uniform(lower_left[2], upper_right[2])
    x, y = rand(unifx), rand(unify)
    return [x, y]
end


"""
Run the experiments. 
"""
function run_experiment()
    """
    The distributions functions we are sampling from. 
    """
    function f(x)
        if norm(x) < 1
            return sin(x[1]*4pi) + cos(x[2]*4pi) + 2
        end
        return 0
    end

    bc(_) = unif_sampler_2d((-1, -1), (1, 1))
    global MHC_ = MHC((x)-> f(x), bc, [0, 0])
    global POINTS = [MHC_() for _ in 1:1000000]
    xs = [point[1] for point in POINTS]
    ys = [point[2] for point in POINTS]
    histogram2d(
        xs, ys,
        c = :vik,
        nbins = 100,
        show_empty_bins = :true
    ) |> display
    return nothing
end

run_experiment()

