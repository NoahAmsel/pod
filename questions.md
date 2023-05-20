# Finite Elements
- right now i have order=1 and degree=2. convergence seems to be second order. what are these parameters
    - here is says to set degree to at least 2*order: https://gridap.github.io/Tutorials/dev/pages/t013_poisson_dev_fe/
    - but i find that setting it to 1 = 1*order i still get second order convergence, just with slightly worse error
- the error of the numerical solutions seems to go to zero at the edge of each cell. is this expected? isn't this a little weird?
- building C is extremely slow... but most of these L2 inner products are actually zero. still i don't see how that can help us avoid doing $O(m^2 n)$ work, where $m$ is the size fo the training data and $n$ is the dimension of the finite elements discretization
    - note that if we use a basis that is actually orthogonal instead of the FE basis, then the whole thing reduces to an SVD. a full SVD would still be $O(m^2 n)$ (since $m < n$) but a truncated SVD could be made MUCH faster using randomized Krylov subspace type algorithms
    - but I'm unsure whether it is possible to extend these^ algorithms to pseudomatrices, where one dimension is a discretization of $L^2$ and the other is $\R^m$
    - wait actually we can build the (sparse) mass matrix to make building C way way faster
    - but it's still numerically bad to actually form C at all. wonder if there's a more direct SVD algorithm with respect to an alternative inner product
- how do you solve the generalized eigenvalue problem (see eq 4.24)
    - the problem is that we have this matrix $M$ where it's easy to take the quadratic form (that's just a single integration using the FE basis) but to do a matrix multiplication, there's no way except taking inner products with one hot vectors
    - dense matrix: matvec: O(n^2), inner product: O(n^2)
    - sparse matrix: matvec: O(n), inner product O(n)
    - what i have: inner product O(n), matvec O(n^2)
    - actually maybe this isn't a problem. because the reason it's cheap to compute the inner product is that there's a sparse matrix defined by the FE discretization. and if I can just access that matrix then i can do matvecs quickly

- for N=5, why does the POD error go to zero for rank 20? the FE space has 32 degrees of freedom, not 20

- make sure to state explicitly all the linear algebra methods. don't fudge it
- try using way more training examples than the rank