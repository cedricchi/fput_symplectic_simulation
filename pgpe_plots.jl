using Plots, LaTeXStrings
## Figure global settings
default(fontfamily="Computer Modern",
    grid=false, frame=:box,
    foreground_color_legend=nothing,
    linewidth=1.5, guidefontsize=10, tickfontsize=10, legendfontsize=10,
    annotationfontsize=10, size=(600, 300))

# ## combined plot with labels
# p1 = plot(x->exp(-x),0,2,legend=false,ann=(:top_right,"(a)"))
# xlabel!(L"x");ylabel!(L"e^{-x}")
# p2 = plot(sin,pi,2pi,line=(:mediumseagreen,5,0.6),legend=false,ann=(:top_right,"(b)"),widen=false) # line properties are color, widt, alpha
# xlabel!(L"t");ylabel!(L"f(t)")
# plot(p1,p2,size=(600,300),leftmargin=.5cm)  # equivalent to layout = @layout[a b]; also if you have to fix margin

## show (on noniform x for convenience)
function plot_ground_state(sim::gauss_quadrature_pgpe_simulation, solg, sold)
    ground_plot = plot(ann=(:top_left, "(g)"))
    hline!(ground_plot, [sim.μ0 / sim.g], label="μ0/g", legendfontsize=8)
    plot!(ground_plot, sim.quadrature_x, abs2.(sim.ϕx * solg[end]), label="ground: t=10.0", legendfontsize=8) # t=10.0 for ground state hardcode here
    plot!(ground_plot, sim.quadrature_x, abs2.(sim.ϕx * solg(1)), label="ground: t=1.0", legendfontsize=8)
    # @info abs2.(sim.ϕx*sold(1))
    plot!(ground_plot, sim.quadrature_x, abs2.(sim.ϕx * sold(1)), label="sim: t=1.0", legendfontsize=8)
    plot!(ground_plot, sim.quadrature_x, abs2.(sim.ϕx * sold[end]), label="sim: t=$(sim.sim_time)", legendfontsize=8)
    xlabel!("x")
    ylabel!(L"|\psi|^2")
    title!("Ground State Density")
    return ground_plot
end

function animate_evolution(ϕx, sold, sim; ground_state_series=nothing, μ0=nothing, outfile="evolution.mp4")
    @info "Building animation..."
    anim = @animate for i in eachindex(sold.t)
        ψ = ϕx * sold.u[i]
        plt = plot(sim.quadrature_x, abs2.(ψ), size=(1000, 500), xlims=(-15, 15), ylim=(0, (μ0 === nothing ? maximum(abs2.(ψ)) * 1.4 : μ0 / sim.g * 1.4)),
            legend=false, xlabel="x", ylabel=L"|\psi(x,t)|^2", title="Time = $(round(sold.t[i]; digits=2))")
        if ground_state_series !== nothing
            plot!(sim.quadrature_x, ground_state_series, color=:black, lw=1, label="Ground state")
        end
        if μ0 !== nothing
            hline!([μ0 / sim.g], linestyle=:dash, color=:red)
        end
        plt
    end
    gif(anim, outfile, fps=5)
end

## N, E, ELin, EInt, μ diagnostics
function plot_diagnostics(pg::gauss_quadrature_pgpe_simulation, sold, N, E, ELin, EInt, μ, ratio)
    ΔN = N / N[1] .- 1
    ΔE = E / E[1] .- 1
    Δμ = (μ .- μ[1]) ./ μ[1]
    ## Plot error for half the simulation time
    sim_t_mid = div(length(sold.t), 2)
    E_N_err_plot = plot(sold.t[1:sim_t_mid], ΔE[1:sim_t_mid], label=L"ΔE(t)/E_0")
    plot!(E_N_err_plot, sold.t[1:sim_t_mid], ΔN[1:sim_t_mid], label=L"ΔN(t)/N_0", xlabel="t",
        ylabel="error", title="Relative Energy & Number Error",
        ann=(:top_center, "(a)"))
    ## Chemical potential is not conserved in dynamics, we plot it for reference
    ## No longer in final plots
    mu_err_plot = plot(sold.t, Δμ, label=L"Δμ(t)", xlabel="t",
        ylabel="Err Δμ", title="Chemical Potential Error",
        ann=(:top_right, "(b)"))
    # if μ0 !== nothing
    #     hline!(p3, [μ0], linestyle=:dash, label="Target μ₀")
    # end
    Eint_vs_Elin_plot = plot(sold.t, ratio, legend=false, xlabel="t",
        ylabel=L"E_{int} / E_{lin}", ann=(:top_center, "(b)"), title="Ratio (E_Int/E_Lin)")
    # t_ratio_mode_plot = surface(sold.t, ratio, abs2.(sold[pg.special_mode_number, :]),
    #     xlabel="t", ylabel="Eint/Elin", zlabel="|mode $(pg.special_mode_number - 1)|^2",
    #     title="(e) Recurrence <-> Ratio", box=:false)

    # Plot a scatter of recurrence vs ratio when recurrence happens
    # ratio_scatter = Float64[]
    # cn2_scatter = Float64[]
    # for i in 1:length(sold.t)
    #     if (pg.n0 * pg.recurrence_ratio_criteria) < abs2.(sold.u[i][pg.special_mode_number])
    #         push!(ratio_scatter, ratio[i])
    #         push!(cn2_scatter, abs2.(sold.u[i][pg.special_mode_number]))
    #     end
    # end
    # recur_ratio_plot = scatter(ratio_scatter, cn2_scatter, xlabel="ratio", ylabel="|c_n(t)|^2",
    #     title="Recurrence (mode $(pg.special_mode_number - 1)>$(Int(pg.recurrence_ratio_criteria*100))%) No.: $(length(cn2_scatter))) vs Ratio",
    #     markercolor=:black, ann=(:top_center, "(f)"), legend=false, alpha=0.6)
    return E_N_err_plot, mu_err_plot, Eint_vs_Elin_plot
end

## plot some accurate modes
function plot_time_evo(sold, pg::gauss_quadrature_pgpe_simulation, λ)
    sim_plot = plot(; title="Modes Evo. λ=$(round(λ, digits=5))",
        yminorgrid=true, legend=:topright,
        xlabel="t", ylabel="|c_n(t)|^2", ann=(:top_center, "(c)"))
    # for m in 1:300 plot!(sim_plot, sold.t, (t -> abs2(sold(t)[m])).(sold.t), legend=false) end
    for m in 1:pg.quadrature_modes
        # @info m (abs2.(sold[m, :]) |> sum) (λ / pg.g)
        if any(abs2.(sold[m, :]) .>= (pg.n0 * 0.1))  # only label modes with significant population
            # plot!(sim_plot, sold.t, (t -> abs2(sold(t)[m])).(sold.t), label="mode: $m", legendfontsize=8)
            plot!(sim_plot, sold.t, abs2.(sold[m, :]), alpha=0.6, label="mode: $(m-1)", legendfontsize=8)
            # else
            # plot!(sim_plot, sold.t, (t -> abs2(sold(t)[m])).(sold.t), label="")
            # plot!(sim_plot, sold.t, 1 .+ abs2.(sold[m, :]), alpha=0.6, label="")
        end
    end
    return sim_plot
end

## heatmap of density
function plot_density_heatmap(sold, pg::gauss_quadrature_pgpe_simulation)
    psix = pg.proj_m_complex * sold[:, :]
    density_heatmap_plot = heatmap(sold.t, pg.quadrature_x, abs2.(psix),
        xlabel="t", ylabel="x", ylims=(-8, 8), ann=(:top_center, text("(d)", :white)),
        title="Heatmap of Density (initial mode $(pg.special_mode_number - 1))")
    return density_heatmap_plot
end

function plot_fourier_transform(freqs, amplitude, pg::gauss_quadrature_pgpe_simulation)
    mask = amplitude .> 10 # zoom in on significant peaks
    plot(freqs[mask], amplitude[mask],
        xlabel="Frequency",
        ylabel="Amplitude",
        xlims=(minimum(freqs[mask]), maximum(freqs[mask])),
        title="Spectrum (mode $(pg.special_mode_number - 1))",
        legend=false, ann=(:top_center, "(e)"))
end