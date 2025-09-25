include("./gauss_quadrature_pgpe_simulation.jl")
using DynamicalSystems
sim = gauss_quadrature_pgpe_simulation(64; g=0.01, sim_time=100.0, sim_time_steps=100, special_mode_number=3, n0=8000, recurrence_ratio_criteria=0.97)
#=
    A Lyapunov exponent quantifies the exponential rate of separation or convergence
    of infinitesimally close trajectories in a dynamical system, with a positive exponent
    indicating exponential divergence and chaos, while a negative exponent signifies
    convergence and stability. In chaotic systems, even tiny differences in initial
    conditions grow exponentially, making long-term prediction impossible. The Lyapunov
    exponent is a fundamental measure of sensitive dependence on initial conditions,
    used to identify chaotic behavior and is a key concept in the study of nonlinear dynamics. 
=#
function pgpe!(pg::gauss_quadrature_pgpe_simulation, dc, c, p, t)
    # c is coefficient vector (modes)
    projected_c_n_t = pg.proj_m_complex * c
    # back to mode-space
    projected_back_c_n_t = pg.Proj_m_hc * (pg.quadrature_w .* abs2.(projected_c_n_t) .* projected_c_n_t) # nonlinear in x
    @. dc = -im * (pg.energy_lvls * c + pg.g * projected_back_c_n_t)         # imaginary time
end
c0 = sim.init_c
# define the ContinuousDynamicalSystem
ds = ContinuousDynamicalSystem(pgpe!, c0; # no explicit jacobian -> autodiff
                                param = nothing)

# compute Lyapunov spectrum
# Choose number of steps, dt between reorthogonalizations, and possibly a transient time to discard
λs = lyapunovspectrum(ds, 100; dt = some_dt, Ttr = some_transient_time)

println("Lyapunov spectrum: ", λs)
println("Maximal Lyapunov exponent: ", maximum(λs))
perturb = 1e-8
c0_perturbed = copy(c0)
c0_perturbed[sim.special_mode_number] += perturb
