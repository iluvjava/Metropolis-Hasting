### We try to use the Metropolis algorithm to implement the Simulated Annealing 
### method and then apply it to combinatorics problems: Knapsack Problem. 

include("../src/core.jl")
using LinearAlgebra, Distributions, Plots, IterTools, ProgressMeter

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

function get_simulated_annealing()
    w = 1                                               # total weight allowed. 
    n = 500                                             # items not part of the solutions!  
    m = 500                                               # items in the solution! 
    global solns = rand(Uniform(0, w), m - 1)|>sort
    insert!(solns, 1, 0)                                # pad the head
    push!(solns, 1)                                     # pad the tail 
    solns = solns[2:end] - solns[1:end - 1]
    not_solns = rand(Uniform(0, w + 1), n)
    weights = vcat(solns, not_solns)                    # the weights for all items. 
    function obj_fxn(x)
        dotted = dot(x, weights)
        if dotted > 1 
            return -Inf
        end
        return dotted
    end
    sim_anea = SA(BSDRW(m + n), (x) -> obj_fxn(x), zeros(Int, m + n), 1)
    return sim_anea
end


function run_experiment1()
    sim_annea = get_simulated_annealing()
    temprs = Vector()
    @showprogress for tempr in (LinRange(0.001, 1, 10) |> collect)[end:-1:1].^2
        change_temp!(sim_annea, tempr)
        for _ in 1:100
            push!(temprs, tempr)
            sim_annea()
        end
        # opt_restart!(sim_annea)
    end
    println("Current best optimal value is: $(sim_annea.opt)")
    fig = plot(sim_annea.obj_values, label="obj val", legend=:right)
    plot!(fig, temprs, label="temperature")
    fig |> display

    return sim_annea
end

function run_experiment2()
    sim_annea = get_simulated_annealing()
    temprs = Vector()
    @showprogress for tempr in exp.(LinRange(0, -10, 10) |> collect)
        change_temp!(sim_annea, tempr)
        for _ in 1:100
            push!(temprs, tempr)
            sim_annea()
        end
        # opt_restart!(sim_annea)
    end
    println("Current best optimal value is: $(sim_annea.opt)")
    fig = plot(sim_annea.obj_values, label="obj val", legend=:right)
    plot!(fig, temprs, label="temperature")
    fig |> display

    return sim_annea
end

function run_experiment3()
    sim_annea = get_simulated_annealing()
    tempr = 1
    k = 1
    previous_opt = sim_annea.opt
    temprs = Vector()
    @showprogress for _ in 1:1000
        sim_annea()
        if sim_annea.opt > previous_opt
            k += 1
            tempr = 1/k
            change_temp!(sim_annea, tempr)
            previous_opt = sim_annea.opt
        end
        push!(temprs, tempr)
    end
    println("Current best optimal value is: $(sim_annea.opt)")
    fig = plot(sim_annea.obj_values, label="obj val", legend=:right)
    plot!(fig, temprs, label="temperature")
    fig |> display
    return sim_annea
end

sa1 = run_experiment1();
sa2 = run_experiment2();
sa3 = run_experiment3();
