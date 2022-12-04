### Sampling circular domain function in a bounded box via MCH with a base chain 
### of Uniform Distributions on the Box. 

include("../src/core.jl")
using Plots, Distributions, LinearAlgebra, IterTools

# directory for saving. 
if !isdir("results")
    mkdir("results")
end

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
function unif_sampler_2d(
    lower_left::Tuple{Real, Real}, 
    upper_right::Tuple{Real, Real}
)
    unifx = Uniform(lower_left[1], upper_right[1])
    unify = Uniform(lower_left[2], upper_right[2])
    x, y = rand(unifx), rand(unify)
    return [x, y]
end

"""
A doubly stochastic chain sampler that uses wrapped guassian distributions on both directions in 2D 
with a fixed variance. 
"""
function wrapped_gaussian_sampler_2d(
    state::Vector{T},
    lower_left::Tuple{Real, Real}, 
    upper_right::Tuple{Real, Real};
    sigma::Real=1
) where {T<:Real}
    lower = [lower_left[1], lower_left[2]]
    upper = [upper_right[1], upper_right[2]]
    function loop_back(x, l, u)  # assert periodic conditions on the rectangle. 
        return mod(x, u - l) + l
    end
    N = Normal(0, sigma)
    return loop_back.(state + rand(N, 2), lower, upper)
end


"""
Run the experiments. 
"""
function run_experiment(
    bc::Function, 
    file_name::String="",
    partition_size=5000,
    partitions_cout=3, 
)
    """
    The distributions functions we are sampling from. 
    """
    function f(x)
        x1 = x[1]
        x2 = x[2]
        if -sin(4pi*x1) + 2sin(2pi*x2)^2 > 1.5
            return sin(x1*4pi) + cos(x2*4pi) + 2
        end
        return 0
    end
    # samples = partition_size*partitions_cout
    samples = 100000
    mhc = MHC((x)-> f(x), bc, [-1/8, 1/8])
    points = [mhc() for _ in 1:samples]|>unique|>collect
    for i in 1:partitions_cout
        xs = [point[1] for point in points[1: i*partition_size]]
        ys = [point[2] for point in points[1: i*partition_size]]
        fig = histogram2d(
            xs, ys,
            c = :acton,
            nbins = 100,
            show_empty_bins = :true
        )
        fig |> display
        if file_name != ""
            savefig(fig, "results/"*file_name*"($(i))")
        end
    end

    return mhc
end

mhc1 = run_experiment("uniform_base") do x
    return unif_sampler_2d((-1, -1), (1, 1))
end
"The rejection rate is: $(mhc1.rejected/mhc1.k)"

mhc2 = run_experiment("gaussian_base") do x
    return wrapped_gaussian_sampler_2d(x, (-1, -1), (1, 1), sigma=0.1)
end
"The rejection rate is: $(mhc2.rejected/mhc2.k)"


