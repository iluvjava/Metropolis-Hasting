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




### ====================================================================================================================
### ISRW: INTEGER RANDOM WALKS BASE CHAIN.
### ====================================================================================================================
"""
ISRW: Integer Sampler with Random Walks. 
------
The doubly stochastic base chain, it's a vector of integers where each element are within a range. 
It samples things by *perturbing each of the element to {1, -1} each with probability 1/4, 
and stay in in the middle with probability 1/2!* . A simple symmetric random walks that is doubly stochastic. 

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
    
    # impose dirichelet boundary conditions for an integers on the range of (l, b), inclusive both ends
    function LoopBackThreshold(x, l, b) 
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
        newstate = LoopBackThreshold.(rand((-1, 1), n).*rand((0, 1), n) + state, l, b)
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
### BSDRW: Binary Single Direction Random walk Base chain 
### ====================================================================================================================




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



### ====================================================================================================================
### The Metroplis Hasting Chain
### ====================================================================================================================

"""
MCH: Metropolis Hasting Chain
------
Given a base chain and a distribution function: f(a prob mass function up to a positive constant)
whose domain include the base chain sampler's range, then metropolis hasting chain carries out the algorithm and try 
to sample from a distribution function. It also requires an initial state too. 

"""
mutable struct MHC
    f::Function                     # target distribution function. 
    bc::Union{Function, BaseChain}  # the base chain instance. 
    x0::Vector                      # the previous state. 
    states::Vector                  # the vector storing all the states of the chain. 
    values::Vector                  # the values for each of the states, stored sparsely. 
    record_interval::Int            # record the states and values every some intervals. 
    approved::Int                   # how many candidates from base chain is approved? 
    rejected::Int                   # how many candidates from the base chain is rejected. 
    k::Int                          # the current iterations numbers we are at for the chain. 

    """
    Set up the MCH using: 
    ### Positional Parameters
    - `f::Function`: a probability assignment function for the base chain. 
    - `bc::BaseChain`: The base chain that we are going to sample our candidates from. 
    - `x0::Vector`: The initial state for this chain as a vector. 
    ### Keyword Parameters
    - `record_interval`: the options to store some of the states every interval and the function 
    value of the states. 
    """
    function MHC(
        f::Function, 
        bc::Union{Function, BaseChain}, 
        x0::Vector; 
        record_interval::Int=typemax(Int)
    )
        this = new() 
        this.f = f
        this.bc = bc 
        this.x0 = x0
        this.states = Vector{typeof(x0)}()
        push!(this.states, x0)
        
        first_val = f(x0)
        this.values = Vector{typeof(first_val)}()
        push!(this.values, first_val)

        this.record_interval = record_interval
        this.rejected = 0
        this.approved = 0
        this.k = 0
        return this
    end

end


"""
Obtain the next sample for the MCH instance. 
"""
function (this::MHC)()
    bc = this.bc; x0 = this.x0; f = this.f
    candidate = bc(x0)
    v = f(x0)
    c = f(candidate)
    r = c/v
    if isnan(r) || isinf(r) || r < 0
        @error("The ratio from the probaility density function is nan, inf, or negative. ")
    end
    this.k += 1             # iteration counter increment. 
    rho = min(c/v, 1)       # acceptance probability of candidate
    next_state = x0         # the next-state.
    next_value = v
    if rand() < rho         # candidate state is approved. 
        this.x0 = candidate
        next_state = candidate  
        next_value = c
        this.approved += 1
    else                    # candidate state is not approved. 
        this.rejected += 1
    end
    # store the information here. 
    push!(this.values, next_value)
    if mod(this.k, this.record_interval) == 0   
        push!(this.states, next_state)
    end
    return next_state
end


"""
Clear all the information stored in this instance and then start fresh. It clears
1. The functions values and all previous states. 
2. The reject and approval rate for the states. 
"""
function empty!(this::MHC)
    this.states = [x0]
    this.values = [f(x0)]
    this.rejected = 0
    this.approved = 0
    this.k = 0
    return this
end



### ============================================================================
### Performs Simulated Annealing for the given objective function and 
### ============================================================================


"""
This struct is the simmulated annealing struct. 
### What it Does
- Formulate the optimization problem in form of MHC. 
- store useful parameters while running the MHC. 
- It's a functor where each call is one sample from the MHC. 
"""
struct SA
    mch::MHC
    bc::Union{BaseChain, Function}
    obj_fxn::Function
    temp::Real

    

end



