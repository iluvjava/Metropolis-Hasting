include("../src/core.jl")
using Plots

f(x) = sqrt(x[1]^2 + x[2]^2) < 100 ? 1 : 0
BCIS_ = BCIS([-100, -100], [100, 100])
MHC_ = MHC(f, BCIS_, [0, 0])
SAMPLES = [MHC_() for I in 1:10000]

