using Gridap.FESpaces: get_algebraic_operator
using LinearAlgebra

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
    C = Symmetric([sum(∫( ψm * ψq )*dΩ) for ψm in training_solutions, ψq in training_solutions])
    C_eigs = eigen(C, (num_train-maximum(ranks)+1):num_train)
    Bs = Dict{Int, Any}(rank => (
        top_eigvecs = C_eigs.vectors[:, end-(rank - 1):end];
        reduce(hcat, [get_free_dof_values(sol) for sol in training_solutions]) * top_eigvecs
    ) for rank in ranks)
    spectrum = reverse(max.(C_eigs.values, 0))
    energy_fractions = Dict{Int, Float64}(
        rank => sum(spectrum[1:rank])/sum(spectrum) for rank in ranks)
    Bs, energy_fractions, spectrum
end

function pod_basis(training_solutions, model_spaces, rank)
    return pod_basis_multirank(training_solutions, model_spaces, [rank])[rank]
end

function apply_pod(operator, B, model_spaces)
    _, U, _, _ = model_spaces
    alg_op = get_algebraic_operator(operator)
    A, f = alg_op.matrix, alg_op.vector
    return FEFunction(U, B * ((B' * A * B) \ (B' * f)))
end

# B = pod_basis(training_solutions, model_spaces, 10)
# new_op = build_circle_op([7, 30], model_spaces)
# new_sol = apply_pod(new_op, B, model_spaces)
# new_sol_hifi = solve(new_op)
# error_fun = new_sol - new_sol_hifi
# sqrt(sum( ∫( error_fun * error_fun )*model_spaces[3] ))
