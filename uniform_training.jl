using Distributions
using Random: MersenneTwister
using Statistics: quantile

include("experiment_utils.jl")
include("pod.jl")

n_train = 200
n_test = 200
discretization_N = 50

model_spaces = build_model_spaces(discretization_N)
training_mus = 10 .^ rand(MersenneTwister(42), Uniform(-1, 1), (2, 2, n_train))
training_solutions = [
    blocks_solution(training_mus[:, :, i], model_spaces)
    for i in 1:size(training_mus)[3]
]

function testing_errors(testing_mus, B, model_spaces)
    [
        (
            mu = testing_mus[:, :, i];
            hifi = blocks_solution(mu, model_spaces);
            pod_approx = apply_pod(build_blocks_op(mu, model_spaces), B, model_spaces);
            error_fun = pod_approx - hifi;
            sqrt(sum( ∫( error_fun * error_fun )*model_spaces[3] ))
        ) for i in 1:size(testing_mus)[3]
    ]
end

function log_ticks(data)
    log_min = floor(log10(minimum(data)))
    log_max = floor(log10(maximum(data)))
    step = max((log_max - log_min) ÷ 4, 1)
    10 .^ [log_min:step:(log_max - step); log_max]
end

left_p = plot(
    xlabel="Basis Rank",
    ylabel="L2 Error",
    yscale=:log10,
    yticks=log_ticks([1e-1, 1e-10]),
    legend=true,
    legendtitle="Training Set Size",
    title="POD Convergence"
)
right_p = twinx(left_p)
plot!(
    right_p,
    ylabel="Fraction of Spectrum's Uncaptured Energy",
    yscale=:log10,
    yticks=log_ticks([1e-2, 1e-16]),
    legend=false
)

for subtrain_size in [10, 20, 50]
    testing_ranks = 1:min(subtrain_size, 50)
    Bs, energy_fractions, spectrum = pod_basis_multirank(training_solutions[1:subtrain_size], model_spaces, testing_ranks)
    testing_mus = 10 .^ rand(MersenneTwister(33), Uniform(-1, 1), (2, 2, n_test))
    error_5p = []
    error_median = []
    error_95p = []
    for rank in testing_ranks
        errors = testing_errors(testing_mus, Bs[rank], model_spaces)
        _5p, _50p, _95p = quantile(errors, [.10, .50, .90])
        push!(error_5p, _5p)
        push!(error_median, _50p)
        push!(error_95p, _95p)
    end
    plot!(left_p, testing_ranks, error_median, ribbon=(error_5p, error_95p), marker=true, label=subtrain_size)
    leftover = Dict(rank => max(1. - ef, eps()) for (rank, ef) in energy_fractions)
    plot!(right_p, leftover, label=subtrain_size, markershape=:x)
end
savefig(left_p, "images/uniform_training/convergence.png")


# new_op = build_circle_op([7, 30], model_spaces)
# new_sol = apply_pod(new_op, B, model_spaces)
# new_sol_hifi = solve(new_op)
# error_fun = new_sol - new_sol_hifi
# sqrt(sum( ∫( error_fun * error_fun )*model_spaces[3] ))

# using LaTeXStrings
# using Plots
# domain_plot(fun; levels=50) = contourf(
#     0:.05:1,
#     0:.05:1,
#     (x, y) -> fun(Point([x, y])),
#     levels=levels,
#     xlabel=L"x",
#     ylabel=L"y",
#     size=(400, 400),
#     # aspect_ratio=:equal
#     dpi=300
# )
# domain_plot(blocks_solution([1 10], 50))
# domain_plot(blocks_solution(reshape([1,10], (:, 1)), 50))
# domain_plot(blocks_solution([0.01 10; 5 5], 50))

