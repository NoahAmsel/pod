using Distributions
using ProgressBars
using Random: MersenneTwister
using SparseArrays
using Statistics: quantile

include("experiment_utils.jl")
include("pod.jl")

#########
discretization_N = 50
train_ns = 50
mu_shape = (2,2)
myrank = 10
#########
model_spaces = build_model_spaces(discretization_N)
V0, U, dΩ, dΓbase = model_spaces
training_mus = mu_set(mu_shape, maximum(train_ns), 42)
training_solutions = [
    blocks_solution(training_mus[:, :, i], model_spaces)
    for i in ProgressBar(1:size(training_mus)[3])
]
B = pod_basis(training_solutions, model_spaces, myrank)
A_basis, f_basis = A_f_basis(mu_shape..., model_spaces)
compressed_A_basis, compressed_f_basis = compress_A_f_basis(B, A_basis, f_basis)
#########
test_mu = mu_set(mu_shape, 33, 33)[:,:,30]
G = build_G(mu_shape, B, model_spaces)
println(residual_op_norm(test_mu, compressed_A_basis, compressed_f_basis, G))
#########
true_u = get_free_dof_values(blocks_solution(test_mu, model_spaces))
test_op = build_blocks_op(test_mu, model_spaces)
a_mu = get_algebraic_operator(test_op).matrix
f_mu = get_algebraic_operator(test_op).vector
pod_u = get_free_dof_values(apply_pod(test_op, B, model_spaces))
r̂ = (pod_u - true_u)' * a_mu
r̂_fun = FEFunction(U, r̂')
println(sqrt(sum(∫(r̂_fun * r̂_fun) * dΩ)))
# norm(a_mu * true_u - f_mu)
#########

Cold = Symmetric([sum(∫( ψm * ψq )*dΩ) for ψm in training_solutions, ψq in training_solutions])

M = mass_matrix(model_spaces)
solution_mat = reduce(hcat, get_free_dof_values(sol) for sol in training_solutions)
Cnew = solution_mat' * (M * solution_mat)