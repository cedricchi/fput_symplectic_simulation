using FastGaussQuadrature, OrdinaryDiffEq
using LinearAlgebra, SparseArrays
using BenchmarkTools, Test, Plots, Plots.Measures, LaTeXStrings
include("./hermite_modes.jl")

## params
g = 1/300
hbar = 1
n = 256         # number of modes in the c-field 

## setup: the cutoff is n-1, giving n modes.
en = (0:n-1) .+ 0.5
x,w,Pxn = gauss_quad(n-1,4)
const Px = Pxn |> complex

## time 
tf = 10.0
nt = 100

# initial state
c0 = rand(ComplexF64, n)
c0[1] = 100.0 + 0im # partially condensed initial
N0 = abs2.(c0) |> sum

function pgpe!(dc,c,p,t)
    ψ = Px*c
    F = Px'*(w.*abs2.(ψ).*ψ)
    @. dc = -im*(en*c + g*F)
end 

function pgpeg!(dc,c,p,t)
    ψ = Px*c
    F = Px'*(w.*abs2.(ψ).*ψ)
    @. dc = -((en .-μ0)*c + g*F)
end

## find ground state
μ0 = 50
prob = ODEProblem(pgpeg!,c0,(0,tf))
@time solg = solve(prob,alg=Vern9(),saveat=range(0,tf,nt+1),abstol=1e-12,reltol=1e-12);
u0(t) = abs2.(ϕx*solg(t))
plot(x,u0(10),size=(800,200),xlims=(-15,15))

## dynamics
cd = solg[end] + randn(ComplexF64,n) # add some noise to ground state
tf = 100.0
nt = 300
ts = range(0,tf,nt+1)
probd = ODEProblem(pgpe!,cd,(0,tf))
@time sold = solve(probd,alg=Vern9(),saveat=ts,abstol=1e-12,reltol=1e-12);

## show (on noniform x for convenience)
ϕx = hermite_modes(x,n-1)
u(t) = abs2.(ϕx*sold(t))
phi(t) = angle.(ϕx*sold(t))
u(1)
plot(x,u(100),size=(800,200),xlims=(-15,15))
plot!(x,u0(10))
hline!([μ0/g])

## animation 
anim = @animate for i in eachindex(sold)
    plot(x,u(ts[i]),size=(800,200),xlims=(-15,15),legend=false)
    plot!(x,u0(10)) 
    hline!([μ0/g])
    ylims!(0,μ0/g*1.4)
end
gif(anim,"therm.gif",fps=12)

## --- Diagnostics ---
# Energy, Number, mu, computed spectrally 
function energy(c)
    ψ = Px*c
    F = Px'*(w.*abs2.(ψ).*ψ)
    dc = @. en*c + 0.5*g*F # factor of 1/2 
    E = sum(conj.(c).*dc) |> real
    return E
end
function mu(c)
    ψ = Px*c
    F = Px'*(w.*abs2.(ψ).*ψ)
    dc = @. en*c + g*F 
    μ = sum(conj.(c).*dc) |> real
    return μ/number(c)
end
number(c) = sum(abs2.(c))

## --- compute energy and number over time ---
function energy_etc(sol)
    ns = length(sol)
    N = zeros(ns)
    E = zeros(ns)
    μ = zeros(ns)
    for i in eachindex(sol)
        N[i] = number(sol[i])
        E[i] = energy(sol[i])
        μ[i] = mu(sol[i])
    end
    return N, E, μ
end
N, E, μ = energy_etc(sold)
ΔN = (N .- N[1])/N[1]
ΔE = (E .- E[1])/E[1]
plot(sold.t,ΔE,label=L"ΔE(t)",ylabel=L"ΔE",xlabel=L"t")
plot(sold.t,ΔN,label=L"ΔN(t)",ylabel=L"ΔN",xlabel=L"t")
plot(sold.t,μ .- μ0,label=L"\mu(t)",ylabel=L"\mu",xlabel=L"t")
μ[end]
μ0

