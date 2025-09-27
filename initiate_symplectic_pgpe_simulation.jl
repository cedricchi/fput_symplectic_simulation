include("./gauss_quadrature_pgpe_simulation.jl")
include("./pgpe_plots.jl")

# (1) create simulation struct
sim = gauss_quadrature_pgpe_simulation(64; g=0.01, sim_time=10000.0, sim_time_steps=100, special_mode_number=3, n0=2000, recurrence_ratio_criteria=0.97)
# (2) set initial state
## start from one excited state, change normalization
sim.init_c[sim.special_mode_number] = sqrt(sim.n0)
λ = sim.g * abs2.(sim.init_c) |> sum
@info "Initial simulation condition: mode $(sim.special_mode_number-1) excited = sqrt(n0)" sim.n0 λ sim.g sim.quadrature_modes sim.sim_time sim.recurrence_ratio_criteria
@time sold = solve_dynamics_IRKGL16!(sim, dt=0.003)
# + 0.1 * randn(ComplexF64, sim.quadrature_modes) # perturbing the last solution slightly with Gaussian noise of size ~0.1 seeding fluctuations
# @time sold = solve_dynamics_vern9!(sim)
# (3) diagnostics and plot
N, E, ELin, EInt, μ, ratio = energy_etc(sim, sold)
E_N_err_plot, mu_err_plot, Eint_vs_Elin_plot = plot_diagnostics(sim, sold, N, E, ELin, EInt, μ, ratio)
sim_plot = plot_time_evo(sold, sim, λ)
density_heatmap_plot = plot_density_heatmap(sold, sim)
freqs, amplitude = fourier_transform(sim, sold)
fft_plot, fft_enlarge = plot_fourier_transform(freqs, amplitude, sim)
plot(E_N_err_plot, Eint_vs_Elin_plot, sim_plot, density_heatmap_plot,
    fft_plot, fft_enlarge, layout=(3, 2), size=(1300, 1000), simd=true)
savefig("n0_$(sim.n0)_t_$(Int(sim.sim_time)).png")
recur_criteria_val = sim.n0 * sim.recurrence_ratio_criteria
nr = count(i -> recur_criteria_val < abs2(sold.u[i][sim.special_mode_number]), 1:length(sold.t)) # count recurrence
er = maximum(ratio)
println("recurrence count: $nr, max ratio: $er")