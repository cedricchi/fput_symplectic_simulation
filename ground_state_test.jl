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
# fast linear algebra
# using MKL # intel

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
# Nc = 40       # lower cutoff for mu=10
Nc = 100        # higher cutoff
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
Px = hermite_modes(xp,Nc)

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
M = 0.001
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

## ========================================
## Energy damping noise in long system limit
# Eq (79) of PHYSICAL REVIEW A 92, 033631 (2015)
#
# Potential is constructed on 4-field grid. 
# Noise is on a 3-field grid.
# however, it is explicitly real valued.
# in the oscillator basis, we can easily construct
# spatially delta-correlated noise from real modes.
# in delta correlated limit, can do in position space.

# TODO noise amplitude units for ω ≂̸ 1

# dimensionless temperature in oscillator units
# T = 0.1

# noise correlation function we _should_ get.
# harmonic oscillator mode δ-function for cutoff n.
# noise power is strongly suppressed near the trap edge.
# physically consitent with large energy of such fluctuations.

function δ(x,y,n) 
    (hermite_modes(x,n)*hermite_modes(y,n)')[1]
end

hermite_modes(.1,n)
hermite_modes(.2,n)'
res = (hermite_modes(.1,n)*hermite_modes(.2,n)')[1]

xp = range(-2*xmax,2*xmax,2Nx)
dtest = δ.(xp,xp',30)

surface(dtest)


## plot some modes 
w = 2
psi0 = hermite_mode.(x,0,w)
plot(x,hermite_mode.(x,0),label="label=Re psi(x), w=1")
plot!(x,real.(psi0),line=(:dash,:red),label="Re psi(x), w=$w")
# plot!(x,imag.(psi0),label="Im")

phi0 = kspace_hermite_mode.(x,0,w)
plot!(x,real.(phi0),line=(:dash,:green),label="Re phi(k), w=$w")
# plot!(x,imag.(phi0),label="Im")

# test norm
dx = diff(x)[1]
@test sum(abs2.(psi0))*dx ≈ 1
@test sum(abs2.(phi0))*dx ≈ 1

## Auxiliary mode projection for arbitrary polynomial with weight exp(-a*x^2/2)
Vt(x,a) = exp(-a*x^2/2)*(1+x-x^2)
a=.1 # frequency of modes to project onto

# 2 field grid for project/reconstruct
x,wx,Px = gauss_quad(2,2,a)
Vtx = Vt.(x,a)      # V on 2-field grid for oscillator modes w=a
d = Px'*(wx.*Vtx);  # coefficients of the w=a modes

Vtr = Px*d
@test Vtr ≈ Vtx

## 2 field grid in kspace for project/reconstruct
k,wk,Pk = kspace_gauss_quad(2,2,a)
Vtk = Pk*d
d2 = Pk'*(wk.*Vtk);  # coefficients of the w=a modes
@test d ≈ d2

## test after change of variables to particular grid 
# e.g. 4 field grid
x,wx,Px = gauss_quad(4,2,a)
b = 3
# Vtr2 = Vt.(x*√b,a)
Vtr2 = Vt.(x/√b,a)

# project onto psi_n^c(x) 
# c = 2*(1-a*b)
c = 2-a/b
xc,wc,Pc = gauss_quad(4,2,c)
d3 = Pc'*(wc.*Vtr2);  # coefficients of the w=c modes
Vtr3 = Pc*d3 
@test Vtr2 ≈ Vtr3

## now take to k-space 
k2,wk2,P2k = kspace_gauss_quad(4,2,c)
Vtk3 = P2k*d3  
d3t = P2k'*(wk2.*Vtk3);  # coefficients of the w=c modes
@test d3 ≈ d3t

## Test on analytic modes of harmonic trap
# should be able to take any mode or product of modes to k-space, and
# 1. back to mode space
# 2. compute norm in k-space 
# 3. do this for terms on any quadrature grid. 
# 4. ?x <-> k from any quad to any other quad?

## first few oscillator modes 
ψ0(x) = exp(-x^2/2)/pi^(1/4)
ψ1(x) = sqrt(2)*x*exp(-x^2/2)/pi^(1/4)
ψ2(x) = (2*x^2-1)*exp(-x^2/2)/pi^(1/4)/2^(1/2)

# transforms
ϕ0(k) = ψ0(k)
ϕ1(k) = -im*ψ1(k)
ϕ2(k) = (-im)^2*ψ2(k)

# test state
c0 = crandn(Nc+1)

## test after change of variables to particular grid 
# e.g. 4 field grid
x,wx,Px = gauss_quad(Nc,4,a)
psi = Px*c0 # 4 field rule

# make a 2 field term and project onto psi_n^c(x) 
# where c is order of new rule
c = 0.5
xc,wc,Pc = gauss_quad(2Nc,2,c)
d0 = Pc'*(wc.*psi.^2);  # coefficients of the w=c modes
# go to kspace 
k,wk,Pk = kspace_gauss_quad(2Nc,2,c)
phi = Pk*d0
# project back
d0i = Pk'*(wk.*phi)
@test d0i ≈ d0

# TODO: compute norm in x and k (should agree according to Parseval)


ϕ(k) = c[1]*ϕ0(k) + c[2]*ϕ1(k) + c[3]*ϕ2(k)

# one field fourier transform on 4 field grid
ϕs(k) = ϕ0(k/sqrt(3))/3^(1/4)

# norm test, and test of scaling, should also be 1
@test quadgk(k->abs2(ϕs(k)),-Inf,Inf)[1] ≈ 1




