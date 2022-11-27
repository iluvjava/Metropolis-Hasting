# The script is create for the final project for STAT 560 class. We will be coding the Metropolis Hasting Algorithm. 

using Distributions


"""
This is a type of struct that represents a doubly stochastic base chain for a Metropolis Hasting algorithm. 
### Traits
--------
- It's a functor. 
--------
### Attributes
- `n::Int`: The dimension of the vector that the base chain accepts for transitioning. 
- `skips::Int`: Do more than one step of sampling. It's carrying out a markov chain using the previous sampled 
state too. 


"""
abstract type BaseChain

end


"""
A Bounded discrete sample descrbes a function whose range is discrete and it's contained in 
boxed in some dimension: n
----
### Attributes
"""
abstract type BoundedDiscreteSampler <: BaseChain

end


"""
ISRW: Integer Sampler with Random Walks. 
------
The doubly stochastic base chain, it's a vector of integers where each element are within a range. 
It samples things by perturbing each of the element to {1, -1} each with probability 1/2. A simple symmetric random walks that can hold. 

-----
It has a periodic boundary conditions on each dimension, this makes it doubly stochastic because uniform distribution 
is the stationary distributions. 

### Attributes
-`len::Int`: The dimension of the vector that we are sampling. 
-`rngl:Vector{Int}`: A vector indicating the low bound for the integer variables on each dimension, inclusive. 
-`rngb:Vector{Int}`: A vector indicating the upper bound the integer random variables on each dimension, inclusive. 
-`skips:Int`: Sample multiple steps of random walks. 
"""
mutable struct ISRW <: BoundedDiscreteSampler
    n::Int
    rngl::Vector{Int}
    rngb::Vector{Int}
    skips::Int

    function ISRW(l::Vector{Int}, b::Vector{Int}, skips::Int=1)
        @assert length(l) == length(b) "The vector passed to BaseChain Integers has to be the same length. "
        @assert all([l[i] <= b[i] for i in 1:length(l)]) "Not all the elements represents an interval. "
        this = new() 
        this.n = length(l)
        this.rngl = l
        this.rngb = b
        this.skips = skips
        return this
    end

    function ISRW(rngs::Vector{Tuple{Int, Int}})
        ISRW([item[1] for item in rngs], [item[2] for item in rngs])
    end

    function ISRW(rng::Tuple{Int, Int})
        return ISRW([rng])
    end

    function ISRW(l::Int, b::Int)
        return ISRW((l, b))
    end

end


function (this::ISRW)(state::Int)
    return this([state])
end


function sample(this::ISRW, state::Vector{Int})
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
        if x < l
            return l - x + l
        end
        if x > b
            return b - x + b
        end
        return x
    end

    # Actual Works
    for _ in 1:this.skips
        newstate = LoopBackThreshold.(rand((-1, 1), n) + state, l, b)
        state = newstate
    end
    return state
end

"""
functor here is just sampling. 
"""
function (this::ISRW)(state::Vector{Int})
    return sample(this, state)
end




### ====================================================================================================================


"""
GPS: Grid Point Sampler 1D. 
------
We make a range with some points in the range, equally spaced with each other and then sample from it with random 
walks type of sampling. 

"""
mutable struct GPS <: BaseChain
    n::Int
    rngl::Vector{Int}
    rngb::Vector{Int}
    bin_counts::Vector{Int}
    skips::Int

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
    bc::Union{Function, BaseChain}
    x0::Vector
    approved::Int
    rejected::Int

    """
    Set up the MCH using: 
    ### Attributes
    - `f::Function`: a probability assignment function for the base chain. 
    - `bc::BaseChain`: The base chain that we are going to sample our candidates from. 
    """
    function MHC(f::Function, bc::Union{Function, BaseChain}, x0::Vector)
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
        this.approved += 1
        return candidate
    end
    this.rejected += 1
    return x0
end