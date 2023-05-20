using ProgressBars
using StatsPlots

include("experiment_utils.jl")
include("pod.jl")

#########
mu_shape = (3, 3)
train_ns = [100]
n_test = 100
discretization_N = 50
testing_ranks = [20, 50]
#########
model_spaces = build_model_spaces(discretization_N)
training_mus = mu_set(mu_shape, maximum(train_ns), 42)
println("Training:")
training_coercs = [exact_coercivity(training_mus[:,:,i], model_spaces) for i in ProgressBar(1:size(training_mus)[3])]
training_solutions = [
    blocks_solution(training_mus[:, :, i], model_spaces)
    for i in ProgressBar(1:size(training_mus)[3])
]

println("Testing:")
testing_mus = mu_set(mu_shape, n_test, 33)
testing_solutions = [
    blocks_solution(testing_mus[:,:,i], model_spaces)
    for i in ProgressBar(1:size(testing_mus)[3])
]

A_basis, f_basis = A_f_basis(mu_shape..., model_spaces)

subtrain_size = 100
Bs, energy_fractions = pod_basis_multirank(training_solutions[1:subtrain_size], model_spaces, testing_ranks)

rank = testing_ranks[2]
####
B = Bs[rank]
errors = testing_errors(testing_mus, testing_solutions, A_basis, f_basis, B, model_spaces)
compressed_A_basis, compressed_f_basis = compress_A_f_basis(B, A_basis, f_basis)
G = build_G(mu_shape, B, model_spaces)

error_upper_bounds = [L2error_bound(testing_mus[:,:,i], compressed_A_basis, compressed_f_basis, G, training_mus, training_coercs) for i in 1:size(testing_mus)[3]]

minimum(error_upper_bounds ./ errors)

#####
test_mu = testing_mus[:,:,1]
pod_approx = apply_pod(test_mu, compressed_A_basis, compressed_f_basis, B, model_spaces);
error_fun = pod_approx - testing_solutions[1];
sqrt(sum( ∫( error_fun * error_fun )*model_spaces[3] ))
error_vec = get_free_dof_values(pod_approx) - get_free_dof_values(testing_solutions[1]);
M = mass_matrix(model_spaces)
sqrt(error_vec' * M * error_vec)
op = get_algebraic_operator(build_blocks_op(test_mu, model_spaces))
A = op.matrix
lhs = A * error_vec
sqrt(lhs' * M * lhs)
residual_op_norm(test_mu, compressed_A_basis, compressed_f_basis, G)

eigvals(Array(A))
result = lobpcg(A, M, false, 1, maxiter=20_000, tol=1e-4)