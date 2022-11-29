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
### ISRW: INTEGER BINOMIAL RANDOM WALKS BASE CHAIN.
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
mutable struct IBRW <: BoundedDiscreteSampler
    n::Int
    rngl::Vector{Int}
    rngb::Vector{Int}
    skips::Int

    function IBRW(l::Vector{Int}, b::Vector{Int}, skips::Int=1)
        @assert length(l) == length(b) "The vector passed to BaseChain Integers has to be the same length. "
        @assert all([l[i] <= b[i] for i in 1:length(l)]) "Not all the elements represents an interval. "
        this = new() 
        this.n = length(l)
        this.rngl = l
        this.rngb = b
        this.skips = skips
        return this
    end

    function IBRW(rngs::Vector{Tuple{Int, Int}})
        IBRW([item[1] for item in rngs], [item[2] for item in rngs])
    end

    function IBRW(rng::Tuple{Int, Int})
        return IBRW([rng])
    end

    function IBRW(l::Int, b::Int)
        return IBRW((l, b))
    end

end


function (this::IBRW)(state::Int)
    return this([state])
end


function sample(this::IBRW, state::Vector{Int})
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
function (this::IBRW)(state::Vector{Int})
    return sample(this, state)
end

### ====================================================================================================================
### BSDRW: Binary Single Direction Random walk Base chain 
### ====================================================================================================================

"""
This is something used a lot for discrete stuff. So let's make a struct so it acts like a functor. 
This type of random walks only changes one element at a time, significantly reducing the variances between each 
transition, making a slow sampling possible. It chooses exactly one element uniformally at random 
and then mutates it, swapping it between {0, 1}
"""
mutable struct BSDRW <: BaseChain
    n::Int
    skips::Int
    function BSDRW(n::Int; skips::Int=1)
        this = new() 
        this.n = n
        this.skips = skips
        return this 
    end
end


"""
sample the next one. 
"""
function (this::BSDRW)(x::Vector{Int})
    @assert all([item in [1, 0] for item in x]) "The state vector pased to BSDRW is non-binary. "
    n = this.n
    idx_mutate = rand(1:n)
    y = copy(x)
    y[idx_mutate] = mod(y[idx_mutate] + 1, 2)
    return y
end

### ====================================================================================================================
### Hyper Cube Uniform Random Walks All Directions 
### ====================================================================================================================

mutable struct HCURWAD
    n::Int
    skips::Int
    l::Vector{Real}
    b::Vector{Real}

end

### ====================================================================================================================
### Hyper Cube Wrapped Gaussian Single Direction Base Chain Sampler
### ====================================================================================================================
"""
A hyper cube with period conditions and we sample from it based on some point using the Guassian Distributions 
but only along a single direction. You can set the variance/std for the single direction guassian sampling 
distributions. Guassian on a periodic interval is known as the Wrapped Guassian. 

### Fields

"""
mutable struct HCWGSDS <: BaseChain 
    n::Int
    skips::Int
    l::Vector{Real}
    b::Vector{Real}
    sigma::Real
    
    """
    l, the lower bound, b is the upper bound. 
    """
    function HCWGSDS(l::Vector, b::Vector, sigma::Real=1; skips::Int=1) 
        this = new()
        @assert length(l) == length(b) "The lower bound and the upper bound for the HCWGSDS has to be the same length. "
        @assert all([l[i] <= b[i] for i in 1:length(l)]) "All elements in l, b has to represent an interval. "
        this.l = l
        this.b = b
        this.sigma = sigma
        this.n = length(l)
        this.skips = skips
        return this
    end


end


"""
Functor for the HCWGSDS, draw one sample from this chain given a state. A wraped guassian with the given 
boundary is drawn. 
"""
function(this::HCWGSDS)(x)
    l = this.l
    b = this.b
    n = this.n
    σ = this.sigma
    @assert (x .>= l .&& x .<= b)|>all "In correct state, one of the element is out of range. "
    i = rand(1:n)
    y = copy(x)
    y[i] = mod(rand(Normal(x[i] + σ)), b[i] - l[i]) + l[i]
    return y
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
    
    mch::MHC                        # the metropolish hasting chain.
    bc::Union{BaseChain, Function}  # the base chain.
    obj_fxn::Function               # objective value of the function.
    f::Function                     # the distributions function.
    
    x_star                          # current found optimal solutions.
    obj_values::Vector              # histocal values for the objective functions.
    distr_values::Vector            # The values of the distributions funcitons.
    
    """
    
    """
    function SA(
        bc::Union{BaseChain, Function}, 
        obj_fxn::Function, 
        x0,
        temp=1
    ) 
        this = new()
        this.mch = mch
        this.bc = bc
        this.obj_fxn = obj_fxn
        this.f = (x) -> exp(obj_fxn(x)/temp)    # define the remapped objective functions. 
        this.mch = MCH(this.f, this.bc, x0)     # defines the inner Metropolis Hasting Chain. 
        this.obj_values = f(x0)                 # The initial value
        this.distr_values = obj_fxn(x0)         # The initial value for the distribution function. 
        return this
    end
    
end

"""
Calls on the function to perform one step of the algorithm. 
"""
function (this::SA)()
    
end


"""
Change the temperature. 
"""
function change_temp!(this::SA, temp::Real)
    @assert temp > 0 "the temperature should be strictly greater than zero but we have $(temp) > 0. "
    this.f = (x) -> exp(obj_fxn(x)/temp)
    return this 
end


"""
Restart the algorithm on the currently found optimal solution, clear data. 
"""
function opt_restart!(this::SA)

end

