### Sampling circular domain function in a bounded box via MCH with a base chain 
### of Uniform Distributions on the Box. 

include("../src/core.jl")
using Plots, Distributions, LinearAlgebra

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
    samples::Int=1000000
)
    """
    The distributions functions we are sampling from. 
    """
    function f(x)
        if norm(x) < 1
            return sin(x[1]*4pi) + cos(x[2]*4pi) + 2
        end
        return 0
    end
    
    mhc = MHC((x)-> f(x), bc, [0, 0])
    points = [mhc() for _ in 1:samples]
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    fig = histogram2d(
        xs, ys,
        c = :grays,
        nbins = 100,
        show_empty_bins = :true
    )
    fig |> display
    if file_name != ""
        savefig(fig, "results/"*file_name)

    end
    return mhc
end

mhc1 = run_experiment("uniform_base", 100000) do x
    return unif_sampler_2d((-1, -1), (1, 1))
end
"The rejection rate is: $(mhc1.rejected/mhc1.k)"

mhc2 = run_experiment("gaussian_base", 100000) do x
    return wrapped_gaussian_sampler_2d(x, (-1, -1), (1, 1), sigma=0.5)
end
"The rejection rate is: $(mhc2.rejected/mhc2.k)"


