using OrdinaryDiffEq, FastGaussQuadrature, IRKGaussLegendre
using LinearAlgebra, FFTW
include("./hermite_modes.jl")

"""
    struct gauss_quadrature_pgpe_simulation

Holds parameters and arrays for a PGPE simulation using Gauss-Hermite quadrature.
Fields are intentionally public for convenience.
"""
mutable struct gauss_quadrature_pgpe_simulation
    quadrature_modes::Int               # cutoff is n-1, giving n modes
    g::Float64                          # interaction strength
    hbar::Float64
    sim_time::Float64
    sim_time_steps::Int
    quadrature_x::Vector{Float64}
    quadrature_w::Vector{Float64}
    proj_m_complex::Array{ComplexF64,2} # Transform matrix, array of all modes for `n=0,1,...,N` with Gaussian weight at `x`
    Proj_m_hc::Array{ComplexF64,2}      # Hermitian conjugate of proj_m_complex
    ϕx::Array{ComplexF64,2}             # Hermite modes evaluated at quadrature points
    energy_lvls::Vector{Float64}        # harmonic oscillator energy levels
    init_c::Vector{ComplexF64}          # partially condensed initial  e.g. init_c[5] = 100.0 + 0im
    special_mode_number::Int            # which mode to excite initially (1-based index)
    n0::Int                             # initial number of particles
    μ0::Float64                         # target chemical potential for imaginary time evolution
    recurrence_ratio_criteria::Float64
end

# constructor helper
function gauss_quadrature_pgpe_simulation(quadrature_modes::Int; g=0.1, hbar=1.0, sim_time=10.0, sim_time_steps=100, special_mode_number=1, n0=2000, μ0=15, recurrence_ratio_criteria=0.97)
    x, w, P = gauss_quad(quadrature_modes - 1, 4)  # P is projection matrix (real) Gauss-Hermite quadrature
    Proj_complex = P |> complex
    Proj_hc = Proj_complex' |> collect      # hermitian clone
    energy_lvls = (0:quadrature_modes-1) .+ 0.5
    init_c = zeros(ComplexF64, quadrature_modes)
    ϕx = hermite_modes(x, quadrature_modes - 1)
    pg = gauss_quadrature_pgpe_simulation(quadrature_modes, g, hbar, sim_time, sim_time_steps, x, w, Proj_complex, Proj_hc, ϕx, energy_lvls, init_c, special_mode_number, n0, μ0, recurrence_ratio_criteria)
    return pg
end

function pgpe!(pg::gauss_quadrature_pgpe_simulation, dc, c, p, t)
    # c is coefficient vector (modes)
    projected_c_n_t = pg.proj_m_complex * c
    # back to mode-space
    projected_back_c_n_t = pg.Proj_m_hc * (pg.quadrature_w .* abs2.(projected_c_n_t) .* projected_c_n_t) # nonlinear in x
    @. dc = -im * (pg.energy_lvls * c + pg.g * projected_back_c_n_t)    # real time evolution
end

function pgpeg!(pg::gauss_quadrature_pgpe_simulation, dc, c, p, t)
    projected_c_n_t = pg.proj_m_complex * c
    projected_back_c_n_t = pg.Proj_m_hc * (pg.quadrature_w .* abs2.(projected_c_n_t) .* projected_c_n_t) # nonlinear in x
    @. dc = -((pg.energy_lvls .- pg.μ0) * c + pg.g * projected_back_c_n_t)  # imaginary time evolution
end

function solve_ground_state!(pg::gauss_quadrature_pgpe_simulation; abstol=1e-12, reltol=1e-12)
    prob = ODEProblem((dc, c, p, t) -> pgpeg!(pg, dc, c, p, t), pg.init_c, (0.0, pg.sim_time))
    return solve(prob, Vern9(), saveat=range(0, pg.sim_time, pg.sim_time_steps + 1), maxiters=1e7, abstol=abstol, reltol=reltol, save_everystep=false, dense=false)
end

# Solve real-time dynamics
function solve_dynamics_vern9!(pg::gauss_quadrature_pgpe_simulation; abstol=1e-12, reltol=1e-12)
    prob = ODEProblem((dc, c, p, t) -> pgpe!(pg, dc, c, p, t), pg.init_c, (0.0, pg.sim_time))
    return solve(prob, Vern9(), saveat=range(0, pg.sim_time, pg.sim_time_steps + 1), maxiters=1e7, abstol=abstol, reltol=reltol, save_everystep=false, dense=false)
end

function solve_dynamics_IRKGL16!(pg::gauss_quadrature_pgpe_simulation; dt=0.03)
    prob = ODEProblem((dc, c, p, t) -> pgpe!(pg, dc, c, p, t), pg.init_c, (0.0, pg.sim_time))
    return solve(prob, alg=IRKGL16(), adaptive=false, abstol=1e-14, reltol=1e-14, dt=dt)
end

## --- Diagnostics ---
# Energy, Number, mu, computed spectrally 
function calc_change(pg::gauss_quadrature_pgpe_simulation, c::Vector{ComplexF64})
    projected_c_n_t = pg.proj_m_complex * c
    projected_back_c_n_t = pg.Proj_m_hc * (pg.quadrature_w .* abs2.(projected_c_n_t) .* projected_c_n_t) # nonlinear in x
    # Total energy
    dE = @. pg.energy_lvls * c + 0.5 * pg.g * projected_back_c_n_t # factor of 1/2
    E = sum(conj.(c) .* dE) |> real
    # Chemical energy
    dμ = @. pg.energy_lvls * c + pg.g * projected_back_c_n_t
    μ = real(sum(conj.(c) .* dμ)) / sum(abs2.(c))
    # Interaction energy
    dEInt = 0.5 * pg.g * projected_back_c_n_t # factor of 1/2 
    EInt = sum(conj.(c) .* dEInt) |> real
    # Linear energy
    dELin = @. pg.energy_lvls * c
    ELin = sum(conj.(c) .* dELin) |> real
    return E, μ, EInt, ELin
end
## --- compute energy and number over time ---
function energy_etc(pg::gauss_quadrature_pgpe_simulation, sol)
    ns = length(sol)
    N = zeros(ns)
    E = zeros(ns)
    μ = zeros(ns)
    EInt = zeros(ns)
    ELin = zeros(ns)
    for i in 1:ns
        ci = sol.u[i]
        N[i] = sum(abs2.(ci))
        E[i], μ[i], EInt[i], ELin[i] = calc_change(pg, ci)
    end
    ratio = EInt ./ ELin
    return N, E, ELin, EInt, μ, ratio
end

function fourier_transform(pg::gauss_quadrature_pgpe_simulation, sold)
    # sold is a solution object, sold.u[i] is the state at time i
    t = sold.t                       # time vector
    y = abs2.(Array(sold)[pg.special_mode_number, :]) # if scalar ODE, yields 1×N array

    # Compute FFT
    Y = fft(y)
    N = length(y)
    Δt = t[2] - t[1]                 # assuming uniform spacing
    fs = 1 / Δt                      # sampling frequency
    # fftfreq(N, Δt)
    freqs = 2π * fftfreq(N, fs)       # frequencies (positive and negative)

    # Compute amplitude spectrum
    amplitude = abs.(Y) / N

    # Plot (only positive frequencies)
    half = 1:div(N, 2)
    return freqs[half], amplitude[half]
end
