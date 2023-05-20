using Distributions
using ProgressBars
using Random: MersenneTwister
using Statistics: quantile

include("experiment_utils.jl")
include("pod.jl")

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

function blocks_random_training(mu_shape, train_ns, n_test, discretization_N, ranks)
    model_spaces = build_model_spaces(discretization_N)
    training_mus = mu_set(mu_shape, maximum(train_ns), 42)
    println("Training:")
    training_solutions = [
        blocks_solution(training_mus[:, :, i], model_spaces)
        for i in ProgressBar(1:size(training_mus)[3])
    ]

    title = latexstring("POD Convergence\n\$N = $discretization_N, \\mu \\in \\left[\\frac{1}{10}, 10\\right]^{$(mu_shape[1]) \\times $(mu_shape[2])} \$")
    left_p = plot(
        xlabel="Basis Rank",
        ylabel="L2 Error",
        yscale=:log10,
        yticks=log_ticks([1e-1, 1e-10]),
        legend=true,
        legendtitle="Training Size",
        title=title,
        dpi=300,
        size=(600, 500)
    )
    right_p = twinx(left_p)
    plot!(
        right_p,
        ylabel="Fraction of Spectrum's Uncaptured Energy",
        yscale=:log10,
        yticks=log_ticks([1e-2, 1e-16]),
        legend=false
    )

    A_basis, f_basis = A_f_basis(mu_shape..., model_spaces)

    println("Testing:")
    testing_mus = mu_set(mu_shape, n_test, 33)
    testing_solutions = [blocks_solution(testing_mus[:,:,i], model_spaces) for i in ProgressBar(1:size(testing_mus)[3])]
    for subtrain_size in train_ns
        testing_ranks = filter(r -> r <= subtrain_size, ranks)
        Bs, energy_fractions = pod_basis_multirank(training_solutions[1:subtrain_size], model_spaces, testing_ranks)        
        error_5p = []
        error_median = []
        error_95p = []
        println("Eval with training size:", subtrain_size)
        for rank in ProgressBar(testing_ranks)
            errors = testing_errors(testing_mus, testing_solutions, A_basis, f_basis, Bs[rank], model_spaces)
            _5p, _50p, _95p = quantile(errors, [.05, .50, .95])
            push!(error_5p, _5p)
            push!(error_median, _50p)
            push!(error_95p, _95p)
        end
        plot!(left_p, testing_ranks, error_median, ribbon=(error_5p, error_95p), marker=true, label=subtrain_size)
        leftover = Dict(rank => max(1. - ef, eps()) for (rank, ef) in energy_fractions)
        plot!(right_p, leftover, label=subtrain_size, markershape=:x)
    end
    return left_p
end

# mu_shape = (3, 3)
# train_ns = reverse([50, 200])
# n_test = 100
# discretization_N = 500
# ranks = 10:10:200
# cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
# p = blocks_random_training(mu_shape, train_ns, n_test, discretization_N, ranks)
# savefig(p, "images/uniform_training/convergence_N500_M200.png")

# mu_shape = (2, 2)
# train_ns = reverse([10, 20, 50])
# n_test = 100
# discretization_N = 50
# ranks = 1:50
# cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
# p = blocks_random_training(mu_shape, train_ns, n_test, discretization_N, ranks)
# savefig(p, "images/uniform_training/convergence_small.png")

# mu_shape = (3, 3)
# train_ns = reverse([25, 50, 100])
# n_test = 100
# discretization_N = 50
# ranks = 1:100
# cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
# p = blocks_random_training(mu_shape, train_ns, n_test, discretization_N, ranks)
# savefig(p, "images/uniform_training/convergence_N50_M100.png")

mu_shape = (3, 3)
train_ns = reverse([10, 25, 50])
n_test = 100
discretization_N = 5
ranks = 1:50
cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
p = blocks_random_training(mu_shape, train_ns, n_test, discretization_N, ranks)
savefig(p, "images/uniform_training/convergence_N5_M50.png")