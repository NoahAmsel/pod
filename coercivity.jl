using ProgressBars
using StatsPlots

include("experiment_utils.jl")
include("pod.jl")

#########
mu_shape = (3, 3)
train_ns = [100, 200, 300, 400]
n_test = 1000
discretization_N = 100
#########
model_spaces = build_model_spaces(discretization_N)
training_mus = mu_set(mu_shape, maximum(train_ns), 42)
println("Training:")
training_coercs = [exact_coercivity(training_mus[:,:,i], model_spaces) for i in ProgressBar(1:size(training_mus)[3])]

println("Testing:")
testing_mus = mu_set(mu_shape, n_test, 33)
test_coercs = [exact_coercivity(testing_mus[:,:,i], model_spaces) for i in ProgressBar(1:size(testing_mus)[3])]

all_ratios = [test_coercs[i] / multi_param_minθ_lower(testing_mus[:,:,i], training_mus[:,:,1:n], training_coercs[1:n]) for n in train_ns for i in 1:size(testing_mus)[3]]
all_trainNs = repeat(string.(train_ns), inner=size(testing_mus)[3])
box_p = violin(all_trainNs, all_ratios, linewidth=0, legend=false, xlabel="Training Size", ylabel="Approximation Ratio of Lower Bound", title="Quality of Lower Bound", dpi=300)
boxplot!(box_p, all_trainNs, all_ratios, fillalpha=0.75, linewidth=2)
cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
savefig("images/coercivity/boxplot.png")

lower100 = [multi_param_minθ_lower(testing_mus[:,:,i], training_mus[:,:,1:minimum(train_ns)], training_coercs[1:minimum(train_ns)]) for i in 1:size(testing_mus)[3]]
p = scatter(
    lower100,
    test_coercs,
    xscale=:log10,
    xlabel="Lower Bound on Coercivity",
    yscale=:log10,
    ylabel="True Coercivity",
    legend=false,
    # yticks=log_ticks(test_coercs),
    yticks=[1/sqrt(10), 1, sqrt(10), 10],
    title="True Coercivity vs. Lower Bound\n(100 training samples)",
    dpi=300
)
plot!(p, [minimum(lower100), maximum(lower100)], [minimum(lower100), maximum(lower100)])
cd("/Users/noah/Documents/PhD1/Numerical Methods 2/Final/code")
savefig("images/coercivity/scatter.png")
