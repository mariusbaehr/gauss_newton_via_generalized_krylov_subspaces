Implementation of my bachelor's thesis based on "An Efficient Implementation of the Gauss–Newton Method Via Generalized Krylow Subspaces" https://doi.org/10.1007/s10915-023-02360-w

Note on not using scipy functions lstqr and lsqr:
The usage of scipy.linalg.lstqr and scipy.sparse.linalg.lsqr were replaced by QR decomposition and CGLS methods, respectively.
Because a discussion of the underlying LAPACK routines would exeed the scope of my bachelor thesis.
