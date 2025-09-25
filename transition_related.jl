include("./gauss_quadrature_pgpe_simulation.jl")
include("./pgpe_plots.jl")

# (1) create simulation struct
sim = gauss_quadrature_pgpe_simulation(64; g=0.01, sim_time=10000.0, sim_time_steps=100, special_mode_number=3, n0=2000, recurrence_ratio_criteria=0.97)
# (2) set initial state
## start from one excited state, change normalization
sim.init_c[sim.special_mode_number] = sqrt(sim.n0)
λ = sim.g * abs2.(sim.init_c) |> sum
@info "Initial simulation condition: mode $(sim.special_mode_number-1) excited = sqrt(n0)" sim.n0 λ sim.g sim.quadrature_modes sim.sim_time
@time sold = solve_dynamics_IRKGL16!(sim, dt=0.003)
# + 0.1 * randn(ComplexF64, sim.quadrature_modes) # perturbing the last solution slightly with Gaussian noise of size ~0.1 seeding fluctuations
# @time sold = solve_dynamics_vern9!(sim)
# (3) diagnostics and plot
N, E, ELin, EInt, μ, ratio = energy_etc(sim, sold)
recur_criteria_val = sim.n0 * sim.recurrence_ratio_criteria
nr = count(i -> recur_criteria_val < abs2(sold.u[i][sim.special_mode_number]), 1:length(sold.t)) # count recurrence
er = maximum(ratio)
println("recurrence count: $nr, max ratio: $er")