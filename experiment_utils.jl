using Distributions: Uniform
using LaTeXStrings
using Plots
using Random: MersenneTwister

function convergence_curve(approx, truth, rs)
    h2error = Dict{Float64, Float64}()
    for r in rs
        N = 2^r
        h = 1/N
        numerical_sol = approx(N)
        error_fun = numerical_sol - truth
        dΩ = Measure(numerical_sol.fe_space.space.fe_basis.trian, 2)
        h2error[h] = sqrt(sum( ∫( error_fun * error_fun )*dΩ ))
    end
    h2error
end

function log_ticks(data)
    log_min = floor(log10(minimum(data)))
    log_max = floor(log10(maximum(data)))
    step = max((log_max - log_min) ÷ 4, 1)
    10 .^ [log_min:step:(log_max - step); log_max]
end

function add_error_line(p, xs, error_fun, N, multiplier=1)
    plot!(p, xs, error_fun(xs, N) * multiplier,
        label=latexstring("N = $N") * (multiplier == 1 ? "" : ", scaled " * latexstring("\\times $multiplier"))
    )
end

domain_plot(fun; levels=50, kwargs...) = contourf(
    0:.05:1,
    0:.05:1,
    (x, y) -> fun(Point([x, y])),
    levels=levels,
    xlabel=L"x",
    ylabel=L"y",
    size=(400, 400),
    # aspect_ratio=:equal
    dpi=300;
    kwargs...
)

# TODO: make this deterministic grid?
mu_set(mu_shape, num, seed) = 10 .^ rand(MersenneTwister(seed), Uniform(-1, 1), (mu_shape..., num))

function testing_errors(testing_mus, testing_solutions, A_basis, f_basis, B, model_spaces)
    @assert size(testing_mus)[3] == length(testing_solutions)
    compressed_A_basis, compressed_f_basis = compress_A_f_basis(B, A_basis, f_basis)
    collect(Float64,
        (
            mu = testing_mus[:, :, i];
            pod_approx = apply_pod(mu, compressed_A_basis, compressed_f_basis, B, model_spaces);
            error_fun = pod_approx - testing_solutions[i];
            sqrt(sum( ∫( error_fun * error_fun )*model_spaces[3] ))
        ) for i in 1:size(testing_mus)[3]
    )
end
