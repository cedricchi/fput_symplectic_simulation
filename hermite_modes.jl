crandn(args...) = randn(ComplexF64,args...)

## hermite modes via stable recursion

"""
    phi_N = hermite_mode(x,N)

Hermite mode with Gaussian weight at `x`. Normalization is 1.
"""
function hermite_mode(x,N)
    Z₀ = 1.0
    Z₁ = √2*x*Z₀
    logB = 0.0
    E = 0.0
    if N == 0 
        return Z₀*exp(-x^2/2)*π^(-1/4)
    elseif N == 1
        return Z₁*exp(-x^2/2)*π^(-1/4)
    else
        Zₙ₊₁,Zₙ,Zₙ₋₁ = zero(x),Z₁,Z₀
        for n in 1:N-1
            Zₙ₊₁ = sqrt(2/(n+1))*x*Zₙ - sqrt(n/(n+1))*Zₙ₋₁/2^E
            E = log(abs(Zₙ₊₁))*(Zₙ₊₁ ≠ 0)
            logB += E
            Zₙ,Zₙ₋₁ = Zₙ₊₁,Zₙ
            Zₙ /= 2^E
        end
    return Zₙ*exp(-x^2/2+log(2)*logB)*π^(-1/4) # undo scaling
    end
end

"""
    m = harmonic_modes(x,N)

Array of all modes for `n=0,1,...,N` with Gaussian weight at `x`.
"""
function hermite_modes(x,N)
    @assert N >= 0 
    l = length(x)
    Z = zeros(l,N+1)
    logB = zeros(l,N+1)
    E = zeros(l)
    Z[:,1] = ones(l)
    if N >= 1
        @. Z[:,2] = √2*x
    end
    if N >= 2
        for n in 1:N-1
            # keep relative scaling the same
            @. Z[:,n+2] = sqrt(2/(n+1))*x*Z[:,n+1] - sqrt(n/(n+1))*Z[:,n]/2^E 
            # accumulate scaling factors 
            @. E = log(abs(Z[:,n+2]))*(Z[:,n+2] != 0)
            @. logB[:,n+2] = logB[:,n+1] + E
            # rescale to avoid overflow 
            @. Z[:,n+2] /= 2^E
        end
    end
    # undo scaling 
    return @. Z *= exp(-x^2/2+log(2)*logB)*π^(-1/4)
end

"""
    rule_order = gauss_quad_rule(k,N)

Quadrature rule order for a k-field product of N fields, where N is the cutoff.
"""
gauss_quad_rule(N,k) = ceil((k*N+1)/2) |> Int

"""
    x,wx,Px = gauss_quad(p,N,ω=1)

Quadrature arrays for a product of `p` fields with cutoff `N`. 
Roots `x` and weights `wx` are used with stable recursion to evaluate the transform matrix `Px` accurately to high order.

Variable change is performed to put the product in suitable form for summing using the weights wx to evaluate the integral of order `x`. This step removes the weight `exp(-ω*x^2)` by change of variables. 
"""
function gauss_quad(N,p=2,ω=1)
    q = gauss_quad_rule(N,p)
    xi,wi = gausshermite(q) 
    # x,wx = sqrt(2/ω/p)*xi,sqrt(2/ω/p)*wi.*exp.(xi.^2)
    x = sqrt(2/ω/p)*xi
    wx = @. 1/hermite_mode(xi,q-1)^2/q*sqrt(2/ω/p)
    Px = hermite_modes(x,N,ω)
    return x,wx,Px
end

"""
    x,wx,Px = gauss_quad(p,N,ω=1)

Solved via diagonalisation of step operator x matrix. Useful to remove deps for e.g. port to MATLAB.
"""
function gauss_quad2(N,p=2,ω=1)
    q = gauss_quad_rule(N,p)
    a,a⁺ = step_operators(n)
    x̄ = (a + a⁺)/√2
    xi = eigs(x̄,nev=n)[1] 
    iseven(n) ? push!(xi,0.) : append!(xi,-xi[end])
    xi = sort(xi)
    x = sqrt(2/ω/p)*xi
    wx = @. 1/hermite_mode(xi,q-1)^2/q*sqrt(2/ω/p)
    Px = hermite_modes(x,N,ω)
    return x,wx,Px
end

"""
    k,wk,Pk = kspace_gauss_quad(N,p,ω=1)

Quadrature arrays for a product of `p` fields with cutoff `N`. 
Roots `k` and weights `wk` are used with stable recursion to evaluate the transform matrix `Pk` accurately to high order.

Variable change is performed to put the product in suitable form for summing using the weights `wk` to evaluate the integral of order `k`. This step removes the weight `exp(-k^2/ω)` by change of variables. 
"""
function kspace_gauss_quad(N,p=2,ω=1)
    ki,wi = gausshermite(gauss_quad_rule(N,p)) 
    k,wk = sqrt(2*ω/p)*ki,sqrt(2*ω/p)*wi.*exp.(ki.^2)
    Pk = kspace_hermite_modes(k,N,ω)
    return k,wk,Pk
end


"""
    psi = hermite_mode(x,N,w)

Single mode. `w` is the oscillator frequency of the mode with quantum number `N`. 
"""
function hermite_mode(x,N,w)
    w^(1/4)*hermite_mode(√w*x,N)
end

"""
    psi = hermite_modes(x,N,w)

Array of all modes up to `N` at `x`, with oscillator frequency `w`.
"""
function hermite_modes(x,N,w)
    w^(1/4)*hermite_modes(√w*x,N)
end

"""
    psi = kspace_hermite_mode(k,N,w)

Single mode. `w` is the oscillator frequency of the mode with quantum number `N`. In `k`-space the width scales inverse to `√w`. 
"""
function kspace_hermite_mode(k,N,w=1)
    w^(-1/4)*hermite_mode(k/√w,N)*(-im)^N
end

"""
    psi = kspace_hermite_modes(k,N,w)

Array of all modes up to `N`. `w` is the oscillator frequency of the mode with quantum number `N`. In `k`-space the width scales inverse to `√w`. 
"""
function kspace_hermite_modes(k,N,w=1)
    w^(-1/4)*hermite_modes(k/√w,N).*(-im).^(0:N)'
end

"""
    a,a⁺ = step_operators(N,w=1)

Step operators for oscillator with frequency `w` and cutoff `N` as sparse matrix.
"""
function step_operators(N,w=1)
    a = diagm(1=>sqrt.(1:N)) |> sparse
    a⁺ = a'
    return a,a⁺ *√w
end  

"""
    ∂x,∂k = spectral_derivates(N,w=1)
    
Spectral derivative operators as sparse matrices, for frequency `w` and cutoff `N`.
"""
function spectral_derivatives(N,w=1)
    a,a⁺ = step_operators(N)
    ∂x = (a - a⁺)/√2 # dimensionless 
    ∂k = im*(a⁺ - a)/√2 
    return ∂x*√w,∂k/√w
end