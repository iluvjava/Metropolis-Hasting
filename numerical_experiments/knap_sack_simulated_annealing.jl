### We try to use the Metropolis algorithm to implement the Simulated Annealing 
### method and then apply it to combinatorics problems: Knapsack Problem. 

include("../src/core.jl")
using LinearAlgebra, Distributions, Plots

"""
Perform the numerical experiment, we make the exponentially scaled objective function 
for the knapsack objective and then we try to sample from it and make a plot of the 
objective values.  
"""
function run_experiment()
    ### Setting up the problem. 
    w = 1                                               # total weight allowed. 
    n = 100                                             # items not part of the solutions!  
    m = 1000                                            # items in the solution! 
    solns = rand(Uniform(0, w), m - 1)|>sort   
    insert!(solns, 1, 0)                                # pad the head
    push!(solns, 1)                                     # pad the tail 
    solns = solns[2:end] - solns[1:end - 1]
    not_solns = rand(Uniform(0, w), n)
    weights = vcat(solns, not_solns)                    # the weights for all items. 
    bcrw = BSDRW(m + n)    # The base chain. 

    function objval(x)
        if dot(weights, x) > w
            return 0  # infeasible
        end
        # exponentially scaled objective function. 
        return exp(dot(weights, x)/0.5)
    end
    global mhc = MHC((x) -> objval(x), bcrw, zeros(Int, m + n))
    global SAMPLED = [mhc() for _ in 1: 1000]
    plot(mhc.values, yaxis=:log) |> display
    max_idx = argmax(mhc.values)
    max_obj_value = dot(weights, SAMPLED[max_idx])
    println("The max obj_value for napsack is: $(max_obj_value)")
    return nothing 
end

function run_simulated_annealing()
    w = 1                                               # total weight allowed. 
    n = 50                                              # items not part of the solutions!  
    m = 100                                              # items in the solution! 
    solns = rand(Uniform(0, w), m - 1)|>sort   
    insert!(solns, 1, 0)                                # pad the head
    push!(solns, 1)                                     # pad the tail 
    solns = solns[2:end] - solns[1:end - 1]
    not_solns = rand(Uniform(0, w), n)
    weights = vcat(solns, not_solns)                    # the weights for all items. 
    function obj_fxn(x)
        dotted = dot(x, weights)
        if dotted > 1 
            return 0
        end
        return dotted
    end
    sim_anea = SA(BSDRW(m + n), (x) -> obj_fxn(x), zeros(Int, m + n), 0.1)
    return sim_anea
end

"""
Run the simulated annealing with a iteration and temperature schedule. 
Each time when the temperature changes, it will take the current optimal found from all previous iterations 
and then start at the point instead.
"""
function run_simulated_annealing_with_temp()
end

# run_experiment()
sim_annea = run_simulated_annealing()
for _ = 1:10000
    sim_annea()
end
println("Current best optimal value is: $(sim_annea.opt)")
