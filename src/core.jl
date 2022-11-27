# The script is create for the final project for STAT 560 class. We will be coding the Metropolis Hasting Algorithm. 

using Distributions


"""
This is a type of struct that represents a doubly stochastic base chain for a Metropolis Hasting algorithm. 
--------
### Attributes
- `n`: The dimension of the vector that the base chain accepts for transitioning. 
"""
abstract type BaseChain

end

"""
BCIS: Base Chain Integers Sampler
------
The doubly stochastic base chain, it's a vector of integers where each element are within a range. 
It samples things by perturbing each of the element to {1, -1} each with probability 1/2. A simple symmetric random walks that can hold. 

-----
It has a periodic boundary conditions on each dimension, this makes it doubly stochastic because uniform distribution 
is the stationary distributions. 

### Attributes
-`len`: The dimension of the vector that we are sampling. 
-`rngl`: A vector indicating the low bound for the integer variables on each dimension, inclusive. 
-`rngb`: A vector indicating the upper bound the integer random variables on each dimension, inclusive. 
"""
mutable struct BCIS <: BaseChain
    n::Int
    rngl::Vector{Int}
    rngb::Vector{Int}

    function BCIS(l::Vector{Int}, b::Vector{Int})
        @assert length(l) == length(b) "The vector passed to BaseChain Integers has to be the same length. "
        @assert all([l[i] <= b[i] for i in 1:length(l)]) "Not all the elements represents an interval. "
        this = new() 
        this.n = length(l)
        this.rngl = l
        this.rngb = b
        return this
    end

    function BCIS(rngs::Vector{Tuple{Int, Int}})
        BCIS([item[1] for item in rngs], [item[2] for item in rngs])
    end

    function BCIS(rng::Tuple{Int, Int})
        return BCIS([rng])
    end

    function BCIS(l::Int, b::Int)
        return BCIS((l, b))
    end

end


function (this::BCIS)(state::Int)
    return this([state])
end

"""
"""
function (this::BCIS)(state::Vector{Int})
    n = this.n
    l = this.rngl
    b = this.rngb
    if state|>length != n
        @error("Can't transition because the given vector is the wrong dimensions")
    end
    if any([state[i] < l[i] || state[i] > b[i]  for i in 1:length(l)])
        @error("One of the elements in the state vector is out of range. ")
    end
    function LoopBackThreshold(x, l, b) # impose periodic conditions for a number.  
        if l == b
            return l
        end
        return mod(x - l, b + 1 - l) + l
    end

    # Actual Works
    return LoopBackThreshold.(rand((-1, 1), n) + state, l, b)

end


"""
MCH: Metropolis Hasting Chain
------
Given a base chain and a distribution function: f(a prob mass function up to a positive constant)
whose domain include the base chain sampler's range, then metropolis hasting chain carries out the algorithm and try 
to sample from a distribution function. It also requires an initial state too. 

"""
mutable struct MHC
    f::Function
    bc::BaseChain
    x0::Vector

    """
    Set up the MCH using: 
    ### Attributes
    - `f::Function`: a probability assignment function for the base chain. 
    - `bc::BaseChain`: The base chain that we are going to sample our candidates from. 
    """
    function MHC(f::Function, bc::BaseChain, x0::Vector)
        @assert bc.n == length(x0) "The base chain dimension doesn't seem to match the dimension of the initial state. "
        this = new() 
        this.f = f
        this.bc = bc 
        this.x0 = x0
        return this
    end

end


"""
Obtain the next sample for the MCH instance. 
"""
function (this::MHC)()
    bc = this.bc; x0 = this.x0
    candidate = bc(x0)
    r = f(candidate)/f(x0)
    if isnan(r) || isinf(r) || r < 0
        @error("The ratio from the probaility density function is nan, inf, or negative. ")
    end

    rho = min(f(candidate)/f(x0), 1)
    if rand() < rho
        this.x0 = candidate
        return candidate
    end
    return x0
end