import numpy as np

def TSVD(V, settings, trunc_idx=None, test=False):
    """Performs Truncated SVD where Truncation parameter is calculated
    via one of two methods:
        1) according to Rossella et al. 2018 (Optimal Reduced space ...).
        2) Alternatively, if trunc_ixd=n (where n is int), choose n modes with
            largest variance
    arguments
        :V - numpy array (n x M)
        :setttings - config for SVD
        :trunc_idx (opt) - index at which to truncate V.
    returns
        :V_trunc - truncated V (n x trunc_idx)
        :U, :s, :W - i.e. V can be factorized as:
                    V = U @ np.diag(s) @ W = U * s @ W
    """
    U, s, W = np.linalg.svd(V, False)


    if settings.SAVE:

        fp_base = settings.get_X_fp().split("/")[-1][1:]

        np.save(settings.INTERMEDIATE_FP + "U" + fp_base, U)
        np.save(settings.INTERMEDIATE_FP + "s" + fp_base, s)
        np.save(settings.INTERMEDIATE_FP + "W"  + fp_base, W)
    #first singular value
    sing_1 = s[0]
    threshold = np.sqrt(sing_1)

    if not trunc_idx:
        trunc_idx = 0 #number of modes to retain
        for sing in s:
            if sing > threshold:
                trunc_idx += 1
        if trunc_idx == 0: #when all singular values are < 1
            trunc_idx = 1
    else:
        assert type(trunc_idx) == int, "trunc_idx must be an integer"

    print("# modes kept: ", trunc_idx)
    U_trunc = U[:, :trunc_idx]
    W_trunc = W[:trunc_idx, :]
    s_trunc = s[:trunc_idx]
    V_trunc = U_trunc * s_trunc @ W_trunc


    if test:
        #1) Check generalized inverses
        V_plus = W.T * (1 / s) @  U.T #Equivalent to W.T @ np.diag(1 / s) @  U.T
        V_plus_trunc =  W_trunc.T * (1 / s_trunc) @  U_trunc.T

        assert np.allclose(V @ V_plus @ V, V), "V_plus should be generalized inverse of V"
        assert np.allclose(V_trunc @ V_plus_trunc @ V_trunc, V_trunc), "V_plus_trunc should be generalized inverse of V_trunc"

        #2) Check both methods to find V_trunc are equivalent
        # Another way to calculate V_trunc is as follows:
        singular = np.zeros_like(s)
        singular[: trunc_idx] = s[: trunc_idx]
        V_trunc2 = U * singular @ W
        assert np.allclose(V_trunc, V_trunc2)


    return V_trunc, U_trunc, s_trunc, W_trunc

def SVD_V_trunc(U, s, W, modes=-1):
    """helper function to calc V_trunc when U, s, W are known"""

    if not modes == -1: #all modes
        U = U[:, :modes]
        W = W[:modes, :]
        s = s[:modes]

    return U * s @ W

def SVD_reconstruction_trunc(input, U, s, W, modes=-1):
    """Performs SVD reconstruction for an input of dimension:
            i) n, returns n
            ii) M x n, returns M x n
            iii) nx x ny x nz, returns nx x ny x nz
            iv) M x nx x ny x nz, returns M x nx x ny x nz"""

    if not modes == -1: #all modes
        U = U[:, :modes]
        W = W[:modes, :]
        s = s[:modes]

    output = SVD_reconstruction(input, U, s, W)

    return output

def SVD_reconstruction(input, U, s, W):
    batched = False
    shpe = input.shape
    #input can be 1D or 3D - if 2D or 4D then this is batched
    if len(shpe) in [2, 4]:
        batched = True
        input = input.reshape((shpe[0], -1)) #flatten
        input = input.T #transpose
    else:
        input = input.flatten()

    V = U * s @ W
    V_plus =  W.T * (1 / s) @  U.T


    output = V @ (V_plus @ input)

    if batched:
        output = output.T

    output = output.reshape(shpe)
    return output
