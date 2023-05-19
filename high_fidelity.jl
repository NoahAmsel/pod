using Gridap

# LEGEND:
# tag_1: lower left corner
# tag_2: lower right corner
# tag_3: upper left corner
# tag_4: upper right corner
# tag_5: bottom edge
# tag_6: top edge
# tag_7: left edge
# tag_8: right edge
# interior
# boundary

function build_model_spaces(N)
    domain = (0,1,0,1)
    partition = (N, N)
    model = CartesianDiscreteModel(domain, partition)

    order = 1
    reffe = ReferenceFE(lagrangian, Float64, order)
    # tag_6 is the top
    V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="tag_6")

    # Dirichlet along top
    U = TrialFESpace(V0, x -> 0)

    degree = 1
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # tag_5 is the bottom
    Γbase = BoundaryTriangulation(model, tags=["tag_5"])
    dΓbase = Measure(Γbase, degree)

    return V0, U, dΩ, dΓbase
end

function build_op(base_directed_grad, kappa, model_spaces)
    V0, U, dΩ, dΓbase = model_spaces

    a(u,v) = ∫( kappa * ∇(v)⊙∇(u) )*dΩ
    b(v) = ∫( base_directed_grad * v )*dΓbase

    AffineFEOperator(a,b,U,V0)
end

solution(base_directed_grad, kappa, model_spaces) = solve(build_op(base_directed_grad, kappa, model_spaces))
solution(base_directed_grad, kappa, N::Integer) = solution(base_directed_grad, kappa, build_model_spaces(N))

function build_circle_problem(mu)
    mu1, mu2 = mu

    # higher kappa is more resistance
    function kappa(xy)
        if (norm(xy .- VectorValue{2, Float64}([1/2, 1/2])) ≤ 1/4)
            return mu1
        else
            return 1.
        end
    end

    base_directed_grad(_) = mu2

    return base_directed_grad, kappa
end

build_circle_op(mu, model_spaces) = build_op(build_circle_problem(mu)..., model_spaces)
circle_solution(mu, model_spaces) = solution(build_circle_problem(mu)..., model_spaces)
circle_solution(mu, N::Integer) = circle_solution(mu, build_model_spaces(N))

function build_blocks_problem(mu)
    # higher kappa is more resistance
    block(x, num_blocks) = if x > 0 Int(ceil(x * num_blocks)) else 1 end
    kappa(xy) = mu[block(1. - xy[2], size(mu)[1]), block(xy[1], size(mu)[2])]
    base_directed_grad(_) = 1.
    return base_directed_grad, kappa
end
build_blocks_op(mu, model_spaces) = build_op(build_blocks_problem(mu)..., model_spaces)
blocks_solution(mu, model_spaces) = solution(build_blocks_problem(mu)..., model_spaces)
blocks_solution(mu, N::Integer) = blocks_solution(mu, build_model_spaces(N))

function A_f_basis(num_y_blocks, num_x_blocks, model_spaces)
    As = []
    fs = []
    fs_constant = nothing
    for i in 1:(num_x_blocks * num_y_blocks)
        mu_basis_vec = reshape(I(num_x_blocks * num_y_blocks)[i,:], (num_y_blocks, num_x_blocks))
        op_basis_vec = get_algebraic_operator(build_blocks_op(mu_basis_vec, model_spaces))
        push!(As, op_basis_vec.matrix)
        push!(fs, zeros(size(op_basis_vec.vector)))
        fs_constant = op_basis_vec.vector
    end
    As = reshape(As, (num_y_blocks, num_x_blocks))[:]
    fs = reshape(fs, (num_y_blocks, num_x_blocks))[:]
    push!(As, 0*I)
    push!(fs, fs_constant)
    As, fs
end

# function qoi(mu, N)
#     sum(∫( solution(mu, N) )*dΓbase)
# end
