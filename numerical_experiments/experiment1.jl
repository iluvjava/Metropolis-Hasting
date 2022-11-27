include("../src/core.jl")
using Plots, LinearAlgebra

# f(x) = sqrt(x[1]^2 + x[2]^2) < 100 ? 1 : 0
function f(x)
    d = norm(x)
    if d < 100
        return sin(x[1]*2pi/50) + cos(x[2]*2pi/50)+ 2.1
    end
    return 0
end

BC = ISRW([-100, -100], [100, 100], 4)
MHC_ = MHC(f, BC, [0, 0])
SAMPLES = [MHC_() for I in 1:1000000]

XS = [item[1] for item in SAMPLES]
YS = [item[2] for item in SAMPLES]

histogram2d(
    XS, YS, seriestype = :histogram2d,
    c = :vik,
    nbins = 100,
    show_empty_bins = :true
)