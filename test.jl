using Test, FastGaussQuadrature, Plots
include("hermite_modes.jl")

# [✓] weight stability issue. 
# [✓] overflow of recursion for large n

n = 765 # just fails 
x,w = gauss_quad(n);
w

# n = 765 # overflows on last element of recursion
n = 2000 # 5000 is no problem
x,w = gauss_quad(n,4)
w
P = hermite_mode.(x,10)
plot(x,P)

P = hermite_modes(x,100)

plot(x,P[:,99:101])

## orthonormality tests
using QuadGK
quadgk(x->hermite_mode(x,10)^2,-Inf,Inf)
quadgk(x->hermite_mode(x,10)*hermite_mode(x,7),-Inf,Inf)

## tests
Nc = 60
@test gauss_quad_rule(Nc,4) == 2Nc+1 # s wave 
@test gauss_quad_rule(Nc,3) == 3Nc/2 + 1 # multiplicative noise 

## exact integral of first few oscillator modes explicitly
ψ0(x) = exp(-x^2/2)/pi^(1/4)
ψ1(x) = sqrt(2)*x*exp(-x^2/2)/pi^(1/4)
ψ2(x) = (2*x^2-1)*exp(-x^2/2)/pi^(1/4)/2^(1/2)

# k space 
ϕ0(k) = ψ0(k)
ϕ1(k) = -im*ψ1(k)
ϕ2(k) = (-im)^2*ψ2(k)

## test quadratures on random state random coeffs
Nc = 2
c = crandn(Nc+1)
ψ(x) = c[1]*ψ0(x) + c[2]*ψ1(x) + c[3]*ψ2(x) 
ϕ(k) = c[1]*ϕ0(k) + c[2]*ϕ1(k) + c[3]*ϕ2(k)

# norm
using QuadGK, LinearAlgebra
@test quadgk(x->ψ0(x)^2,-Inf,Inf)[1] ≈ 1
@test quadgk(x->ψ1(x)^2,-Inf,Inf)[1] ≈ 1
@test quadgk(x->ψ2(x)^2,-Inf,Inf)[1] ≈ 1

## quadrature rules
## 1-field
x,w,P = gauss_quad(Nc,1)
psi = P*c
@test dot(w,psi) ≈ quadgk(x->ψ(x),-Inf,Inf)[1]

## 2-field
x,w,P = gauss_quad(Nc,2)
psi = P*c
@test dot(w,psi.^2) ≈ quadgk(x->ψ(x)^2,-Inf,Inf)[1]

## 3-field
x,w,P = gauss_quad(Nc,3)
psi = P*c
@test dot(w,psi.^3) ≈ quadgk(x->ψ(x)^3,-Inf,Inf)[1]

## 4-field
x,w,P = gauss_quad(Nc,4)
psi = P*c
@test dot(w,psi.^4) ≈ quadgk(x->ψ(x)^4,-Inf,Inf)[1]
 
## 5-field
# x,w,P = gauss_quad(5,Nc)
# psi = P*c
# @test dot(w,psi.^5) ≈ quadgk


## connect p field rule to r field rule 
# p=4
# x1,w1,P1 = gauss_quad_fields(p,Nc)

# q = 3
# a = 2(1-q/p)
# # xk,wk = gausshermite(quad_rule(k,Nc)) 
# # x,w = sqrt(2/k)*xk,sqrt(2/k)*wk.*exp.(xk.^2)
# P2 = hermite_modes(x1*sqrt(p/a),Nc)
# T1 = P2'.*w1'
# # x2,w2,P2 = gauss_quad_fields(p/(p-1),Nc)


# x3,w3,P3 = gauss_quad_fields(3,Nc)
# ct = crandn(Nc+1)
# T1*(P1*ct)

# xa,wa,Pa = gauss_quad_fields(3,Nc)
# Pa*ct

# function gauss_basis_connection(k,N)
#     xk,wk = gausshermite(quad_rule(k,N)) 
#     x,w = sqrt(2/k)*xk,sqrt(2/k)*wk.*exp.(xk.^2)
#     P = hermite_modes(x,N)
#     return x,w,P
# end