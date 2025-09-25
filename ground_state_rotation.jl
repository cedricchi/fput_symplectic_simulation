using FastGaussQuadrature
using FastTransforms
using QuadGK
using BenchmarkTools
using OrdinaryDiffEq
using LinearAlgebra
using SparseArrays
using Test
using Plots, Plots.Measures
using LaTeXStrings

## modes
include("hermite_modes.jl")

## orthonormality tests
quadgk(x->hermite_mode(x,5)^2,-Inf,Inf)
@time quadgk(x->hermite_mode(x,3)*hermite_mode(x,6),-Inf,Inf)

## make modes for plotting
n = 100
xmax = 1.2*sqrt(2*n+1)
Nx = 20n
x = LinRange(-xmax,xmax,Nx)
dx = x[2]-x[1]
hmat = hermite_modes(x,n);
plot(x,hmat .+ one.(x)*(0.5:n+.5)',legend=false,ylims=(0,n+2))
plot!(x,x.^2/2,line=(:red,0.3,4))
xlabel!(L"x")
ylabel!(L"\psi_n(x)")

## orthonormality
orth1 = hmat'*hmat*dx
heatmap(1 .+ log10.(abs.(orth1)))

## PGPE in 1D
Nc = 40       # lower cutoff for mu=10
# Nc = 100        # higher cutoff  
en = range(0,Nc,Nc+1) .+0.5

## quadrature rule for phi^4
# order of term in Hamiltonian sets order of rule
xk,wk,Pnk = gauss_quad(Nc,4)

Pnk = Pnk |> complex

## PGPE
c0 = crandn(Nc+1)

## 1. compute the vector Ψ
Ψ = Pnk*c0

## 2. compute nonlinearity
Ξ = @. wk*abs2(Ψ)*Ψ

## 3. Compute the projection onto modes
F = Pnk'*Ξ

## params
μ = 10.0
λ = 0.1
tf = 80.0

p = Ψ,Ξ,F,Pnk

## evaluate pgpe in imaginary time to find gnd state
function pgpe!(dc,c,p,t)
    Ψ,Ξ,F,Pnk = p
    mul!(Ψ,Pnk,c)
    @. Ξ = wk*abs2(Ψ)*Ψ
    mul!(F,Pnk',Ξ)
    # real time
    # @. dc = -im*(en*c + λ*F)
    # imag time
    @. dc = -((en-μ)*c + λ*F)
end

prob = ODEProblem(pgpe!,c0,(0,tf),p)

@time sol = solve(prob,alg=Vern6());

##
function plotsol(j;sol=sol)
plot(abs2.(sol[j]),seriestype=:stem,m=:circle,legend=false)
end

plotsol(length(sol.t))

## show ψ
xmax = sqrt(2*Nc+1)
Nx = 200
xp = range(-xmax,xmax,Nx)

# Make const
const Px = hermite_modes(xp,Nc)

function make_psi(Px,c)
    return Px*c
end

ψx = make_psi(Px,sol[length(sol.t)])
plot(xp,abs2.(ψx),label=L"|\psi|^2")
plot!(xp,angle.(ψx),label=L"\textrm{angle}(\psi)")

function show_psi(j;x=xp,sol=sol)
    ψx = make_psi(Px,sol[j])
    plot(x,abs2.(ψx),ylims=(-10,1.2*μ/λ),label=L"|\psi|^2",right_margin=0.6cm,legend=:topleft)
    plot!(twinx(),x,angle.(ψx),ylims=(-pi,pi),label=L"\textrm{angle}(\psi)",c=:red)
end

anim = @animate for j in 1:length(sol.t)
    show_psi(j)
end
gif(anim)

## energy damping potential
# Ve(x) = -M∂x*j(x)

∂x,∂k = spectral_derivatives(Nc)

# test derivatives
dc1 = ∂x*[1;zeros(Nc)]

## Energy damping potential: initialize cache fields of correct size
# 1. second derivative
d0 = ∂x*(∂x*c0)

# 2. map to 4-field grid
psi0 = Pnk*c0
∂xxψ0 = Pnk*d0

# 3. weighted potential term
U0 = @. wk*imag(conj(psi0)*∂xxψ0)*psi0

# 4. project with gauss-hermite quadrature
e0 = Pnk'*U0

## dynamics
const M = 0.001
μ = 10.0
λ = 0.1

# energy damped PGPE
function epgpe(c,p,t)
    ψ = Pnk*c
    # TODO for convoluion jx(x) -> k jx(k)G(k) -> Ve(x)
    ∂xxψ = Pnk*∂x*(∂x*c)
    U = @. wk*imag(conj(ψ)*∂xxψ)*ψ
    V = Pnk'*U

    Ξ = @. wk*abs2(Ψ)*Ψ
    F = Pnk'*Ξ

    # real time
    # @. dc = -im*(en*c + λ*F)
    #imag time
    # @. dc = -((en-μ)*c + λ*F)

    @. -im*((en-μ)*c + λ*F - M*V)
end

## run and animate
tf = 1000
Nt = 100
prob2 = ODEProblem(epgpe,c0,(0,tf),p)

@time sol2 = solve(prob2,alg=Vern6(),saveat=range(0,tf,Nt));

# show 
function show_psi(j;x=xp,sol=sol)
    ψx = make_psi(Px,sol[j])
    plot(x,λ*abs2.(ψx),ylims=(-10,1.2*μ),label=L"|\psi|^2",right_margin=0.6cm,legend=:topleft)
    plot!(twinx(),x,angle.(ψx),ylims=(-pi,pi),label=L"\textrm{angle}(\psi)",c=:red)
    hline!([μ])
end

anim2 = @animate for j in 1:length(sol2.t)
    show_psi(j,sol=sol2)
end
gif(anim2)

