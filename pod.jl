using Gridap.FESpaces: get_algebraic_operator
using IterativeSolvers: lobpcg
using LinearAlgebra
using ProgressBars

include("high_fidelity.jl")

geom_range(start, stop, n) = 10 .^(range(log10(start), log10(stop), n))

# mu1_range = (.1, 10)
# mu2_range = (.01, 100)

# N = 50
# model_spaces = build_model_spaces(N)
# training_solutions = [
#     circle_solution([mu1, mu2], model_spaces)
#     for mu1 in geom_range(mu1_range[1], mu1_range[2], 20)
#     for mu2 = geom_range(mu2_range[1], mu2_range[2], 10)
# ]

function pod_basis_multirank(training_solutions, model_spaces, ranks)
    V0, U, dΩ, dΓbase = model_spaces
    num_train = length(training_solutions)
    println("Building C:")
    M = mass_matrix(model_spaces)
    solution_mat = reduce(hcat, get_free_dof_values(sol) for sol in training_solutions)
    C = Symmetric(solution_mat' * (M * solution_mat))
    println("Taking SVD:")
    C_eigs = eigen(C, (num_train-maximum(ranks)+1):num_train)
    println("Building Bs")
    Bs = Dict{Int, Any}(rank => (
        top_eigvecs = C_eigs.vectors[:, end-(rank - 1):end];
        reduce(hcat, [get_free_dof_values(sol) for sol in training_solutions]) * top_eigvecs
    ) for rank in ProgressBar(ranks))
    spectrum = reverse(max.(C_eigs.values, 0))
    energy_fractions = Dict{Int, Float64}(
        rank => sum(spectrum[1:rank])/sum(spectrum) for rank in ranks)
    Bs, energy_fractions
end

function pod_basis(training_solutions, model_spaces, rank)
    return pod_basis_multirank(training_solutions, model_spaces, [rank])[1][rank]
end

function compress_A_f_basis(B, A_basis, f_basis)
    @assert size(A_basis) == size(f_basis)
    num_y_blocks, num_x_blocks = size(A_basis)
    (
        [B' * A_basis[y, x] * B for x in 1:num_x_blocks for y in 1:num_y_blocks],
        [B' * f_basis[y, x] for x in 1:num_x_blocks for y in 1:num_y_blocks]
    )
end

function compress_A_f_basis(B, A_basis, f_basis)
    (
        [B' * A_basis[i] * B for i in 1:length(A_basis)],
        [B' * f_basis[i] for i in 1:length(A_basis)]
    )
end

function pod_compressed_solve(mu, compressed_A_basis, compressed_f_basis)
    A = sum(compressed_A_basis .* [mu[:]; 1.])
    f = sum(compressed_f_basis .* [mu[:]; 1.])
    A \ f    
end

function apply_pod(mu, compressed_A_basis, compressed_f_basis, B, model_spaces)
    _, U, _, _ = model_spaces
    u_coeffs = pod_compressed_solve(mu, compressed_A_basis, compressed_f_basis)
    return FEFunction(U, B * u_coeffs)
end

function apply_pod(operator, B, model_spaces)
    _, U, _, _ = model_spaces
    alg_op = get_algebraic_operator(operator)
    A, f = alg_op.matrix, alg_op.vector
    return FEFunction(U, B * ((B' * A * B) \ (B' * f)))
end

function build_G(mu, B, model_spaces)
    _, U, dΩ, _ = model_spaces
    A_basis, f_basis = A_f_basis(size(mu)..., model_spaces)
    # TODO: below can be written as a matrix instead of vector of vectors
    # TODO: in our case Aq is symmetric, but properly shouldn't it be ξ' * Aq?
    R_list = [f_basis; [Aq * ξ for Aq in A_basis for ξ in eachcol(B)]]
    R = sparse(reduce(hcat, R_list))  # TODO: is this actually worth it?
    M = mass_matrix(model_spaces)
    R' * M * R
end

function residual_op_norm(mu, compressed_A_basis, compressed_f_basis, G)
    u_pod_reduced_basis = pod_compressed_solve(mu, compressed_A_basis, compressed_f_basis)
    affine_mus = [mu[:]; 1.]
    little_r = [affine_mus; -kron(affine_mus, u_pod_reduced_basis)]
    little_r' * (G * little_r)
end

function exact_coercivity(mu, model_spaces)
    test_op = build_blocks_op(mu, model_spaces)
    a_mu = get_algebraic_operator(test_op).matrix
    M = mass_matrix(model_spaces)
    result = lobpcg(a_mu, M, false, 1, maxiter=20_000, tol=1e-2)
    @assert result.converged
    result.λ[1]
end

minθ_lower(mu, mu_prime, coerc_prime) = coerc_prime * min(minimum(mu ./ mu_prime), 1)
function multi_param_minθ_lower(mu, train_mus, train_coercs)
    @assert size(train_mus)[3] == length(train_coercs)
    maximum(minθ_lower(mu, train_mus[:,:,i], train_coercs[i]) for i in 1:length(train_coercs))
end
