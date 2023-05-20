using Printf: @sprintf
using ProgressBars

include("experiment_utils.jl")
include("pod.jl")


########
function one_test(mu_shape, train_ns, discretization_N, ranks)
    model_spaces = build_model_spaces(discretization_N)
    V0, U, dΩ, dΓbase = model_spaces
    training_mus = mu_set(mu_shape, maximum(train_ns), 42)
    println("Training:")
    training_solutions = [
        blocks_solution(training_mus[:, :, i], model_spaces)
        for i in ProgressBar(1:size(training_mus)[3])
    ]

    Bs, energy_fractions = pod_basis_multirank(training_solutions, model_spaces, ranks)
    ########
    # energy_fraction = [energy_fractions[1]; diff([energy_fractions[rank] for rank in ranks])]
    basis_plots = [domain_plot(FEFunction(U, b), levels=30, title="Basis $i\nEnergy: $(@sprintf("%.3f", 100*energy_fractions[i]))") for (i, b) in enumerate(Iterators.reverse(eachcol(Bs[10])))]
    ########
    testing_mu = mu_set(mu_shape, 2, 33)[:,:,2]
    test_true = blocks_solution(testing_mu, model_spaces)
    testing_operator = build_blocks_op(testing_mu, model_spaces)
    pod_plots = [
        domain_plot(
            apply_pod(testing_operator, Bs[rank], model_spaces), levels=30, title="Solution $rank")
        for rank in ranks
    ]
    error_plots = [
        (test_pod = apply_pod(testing_operator, Bs[rank], model_spaces);
        err = test_pod - test_true;
        domain_plot(err, levels=30, title="Error $rank:\n$(@sprintf("%.4f", sqrt(sum(∫(err*err)*dΩ))))")) # right_margin=2Plots.mm
        for rank in ranks
    ]

    multi_plot = collect(Iterators.flatten(zip(basis_plots, pod_plots, error_plots)))
    first5 = plot(
        multi_plot[1:15]...,
        layout=(5, 3),
        size=(1300, 1600),
    )
    next5 = plot(
        multi_plot[16:end]...,
        layout=(5, 3),
        size=(1300, 1600),
    )
    return first5, next5
end

mu_shape = (3, 3)
train_ns = 100
discretization_N = 50
ranks = 1:10
first5, next5 = one_test(mu_shape, train_ns, discretization_N, ranks)
cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
savefig(first5, "images/one_test/N50_M100_first.png")
savefig(next5, "images/one_test/N50_M100_next.png")

mu_shape = (2, 2)
train_ns = 500
discretization_N = 100
ranks = 1:10
first5, next5 = one_test(mu_shape, train_ns, discretization_N, ranks)
cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
savefig(first5, "images/one_test/N50_M500_22_first.png")
savefig(next5, "images/one_test/N50_M500_22_next.png")