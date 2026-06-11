import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

def get_sparse_matrix(coeffs):
    """Function to return a sparse matrix representation of a set of scalar coefficients"""
    ncv = coeffs.ncv
    data = np.zeros(3*ncv-2)
    rows = np.zeros(3*ncv-2, dtype=int)
    cols = np.zeros(3*ncv-2, dtype=int)
    data[0] = coeffs.aP[0]
    data[1] = coeffs.aE[0]
    rows[0] = 0
    cols[0] = 0
    rows[1] = 0
    cols[1] = 1
    for i in range(ncv-2):
        data[3*i+2] = coeffs.aW[i+1]
        data[3*i+3] = coeffs.aP[i+1]
        data[3*i+4] = coeffs.aE[i+1]
        rows[3*i+2:3*i+5] = i+1
        cols[3*i+2] = i
        cols[3*i+3] = i+1
        cols[3*i+4] = i+2
    data[3*ncv-4] = coeffs.aW[-1]
    data[3*ncv-3] = coeffs.aP[-1]
    rows[3*ncv-4:3*ncv-2] = ncv-1
    cols[3*ncv-4] = ncv-2
    cols[3*ncv-3] = ncv-1
    
    return csr_matrix((data, (rows, cols)))

def solve(coeffs):
    """Function to solve the linear system and return the correction field"""
    # Single scalar equation
    if len(coeffs) == 1:
        A = get_sparse_matrix(coeffs[0])
        b = -coeffs[0].rP

        return spsolve(A, b)
        
    # Coupled pressure-velocity equation
    # solve(PP_coeffs, PU_coeffs, UP_coeffs, UU_coeffs)
    
    elif len(coeffs) == 4:
        PP_coeffs, PU_coeffs, UP_coeffs, UU_coeffs = coeffs

        ncv = PP_coeffs.ncv

        # Build submatrices
        A_PP = get_sparse_matrix(PP_coeffs)
        A_PU = get_sparse_matrix(PU_coeffs)
        A_UP = get_sparse_matrix(UP_coeffs)
        A_UU = get_sparse_matrix(UU_coeffs)

        # Build block matrix
        A = bmat(
            [
                [A_PP, A_PU],
                [A_UP, A_UU],
            ],
            format="csr"
        )

        # Build coupled RHS
        b_P = -(PP_coeffs.rP + PU_coeffs.rP)
        b_U = -(UP_coeffs.rP + UU_coeffs.rP)

        b = np.concatenate([b_P, b_U])

        # Solve coupled system
        dPhi = spsolve(A, b)

        # Split correction vector
        dP = dPhi[:ncv]
        dU = dPhi[ncv:]

        return dP, dU

    else:
        raise ValueError(
            "solve() accepts either one coeffs object or four coeffs objects: "
        )