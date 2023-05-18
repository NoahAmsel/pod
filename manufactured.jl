include("experiment_utils.jl")
include("high_fidelity.jl")

####################
# Solver convergence
####################
# Set up
manufactured_grad(xy) = 2π * cos(2π * xy[1])
manufactured_numerical_sol(N) = solution(manufactured_grad, xy -> 1, N)
manufactured_true_sol(x, y) = cos(2π * x) * sinh(2π * (1 - y)) / cosh(2π)
manufactured_true_sol(xy) = manufactured_true_sol(xy[1], xy[2])

# Full domain
p_true = domain_plot(manufactured_true_sol)
plot!(p_true, title=latexstring("Analytic Solution:\n\$\\cos(2\\pi x) \\cdot \\frac{\\sinh(2\\pi (1-y))}{\\cosh(2\\pi)}\$"))  # u(x,y) = \cos(2\pi x) \sinh(2\pi (1-y))/ \cosh(2\pi)
savefig(p_true, "images/manufactured/true_solution.png")

p_error = domain_plot(manufactured_numerical_sol(20) - manufactured_true_sol, levels=20)
plot!(p_error, title=L"Error of Numerical Solution: $N=20$")
savefig(p_error, "images/manufactured/error20.png")

# Convergence: bottom edge
bottom_error(xs, N) = (manufactured_numerical_sol(N) - manufactured_true_sol).(Point([x, 0]) for x in xs)
p_bottom = plot(
    xlabel=L"x",
    ylabel=L"Error at $u(x, 0)$",
    title="Error of Numerical Solution at Bottom Edge",
    dpi=300
)
add_error_line(p_bottom, 0:0.001:1, bottom_error, 8)
add_error_line(p_bottom, 0:0.001:1, bottom_error, 16)
add_error_line(p_bottom, 0:0.001:1, bottom_error, 32)
add_error_line(p_bottom, 0:0.001:1, bottom_error, 32, (32÷8)^2)
savefig(p_bottom, "images/manufactured/error_bottom_convergence.png")

# Global convergence
convergence_dict = convergence_curve(manufactured_numerical_sol, manufactured_true_sol, 3:10)
p_manufactured_convergence = plot(
    convergence_dict,
    label="FE Solution",
    marker=true,
    xflip=true,
    xscale=:log10,
    xlabel=L"h",
    xticks=log_ticks(keys(convergence_dict)),
    yscale=:log10,
    ylabel="L2 error",
    yticks=log_ticks(values(convergence_dict)),
    title="Convergence of FE Solution",
    dpi=300
)
plot!(
    p_manufactured_convergence,
    collect(keys(convergence_dict)),
    collect(keys(convergence_dict)) .^ 2,
    label=L"h^2"
)
savefig(p_manufactured_convergence, "images/manufactured/global_convergence.png")
