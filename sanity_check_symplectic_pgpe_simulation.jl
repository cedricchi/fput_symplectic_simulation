include("./gauss_quadrature_pgpe_simulation.jl")
include("./pgpe_plots.jl")

# (1) create simulation struct
sim = gauss_quadrature_pgpe_simulation(300; g=0.01, sim_time=10.0, sim_time_steps=100, special_mode_number=1, μ0=15)
# (2) set initial state
sim.init_c = rand(ComplexF64, sim.quadrature_modes)
sim.init_c[sim.special_mode_number] = 100.0 + 0im # make a big condensate
@info "Initial condition - ground state: mode 1 has big condensate 100.0 + 0im" (abs2.(sim.init_c) |> sum) sim.g sim.quadrature_modes sim.sim_time sim.μ0
# sim.init_c = randn(sim.quadrature_modes) + im * randn(sim.quadrature_modes)
# (3) compute ground state
@time solg = solve_ground_state!(sim)
# (4) add noise and run real-time dynamics
# sim.init_c = solg[end]
sim.init_c = solg[end] + 0.1 * randn(ComplexF64, sim.quadrature_modes) # perturbing the last solution slightly with Gaussian noise of size ~0.1 seeding fluctuations
sim.sim_time = 1000.0
sim.sim_time_steps = 100
# sim.init_c = zero.(solg[end])
# sim.special_mode_number = 3
# sim.init_c[sim.special_mode_number] = sqrt(n0)
λ = sim.g * abs2.(sim.init_c) |> sum
@info "Initial condition - dynamic: " λ sim.g sim.quadrature_modes sim.sim_time
# @info "Initial condition:" sim.g sim.quadrature_modes sim.sim_time sim.special_mode_number sim.init_c[sim.special_mode_number]
# @time sold = solve_dynamics_IRKGL16!(sim, dt=0.01)
@time sold = solve_dynamics_vern9!(sim)
# (5) diagnostics and plot
# plot ground state
ground_plot = plot_ground_state(sim, solg, sold)
N, E, ELin, EInt, μ, ratio = energy_etc(sim, sold)
E_N_err_plot, mu_err_plot, Eint_vs_Elin_plot = plot_diagnostics(sim, sold, N, E, ELin, EInt, μ, ratio)
sim_plot = plot_time_evo(sold, sim, λ)
density_heatmap_plot = plot_density_heatmap(sold, sim)
# (6) animate evolution and save gif/mp4
plot(ground_plot, E_N_err_plot, Eint_vs_Elin_plot, sim_plot, density_heatmap_plot, layout=(3, 2), size=(1300, 1000))
savefig("sanity_check.png")
# display(ground_plot)
# display(p1); display(p2); display(p3); display(p4)
