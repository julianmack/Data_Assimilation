import numpy as np
import torch


def cost_fn_J(w, data, settings):
    """Computes VarDA cost function.
    """

    device = data.get("device")
    d = data.get("d")
    G = data.get("G")
    V_trunc = data.get("V_trunc")
    V =  V_trunc if V_trunc is not None else data.get("V")
    V_grad = data.get("V_grad")
    R_inv = data.get("R_inv")

    sigma_2 = settings.OBS_VARIANCE
    alpha = settings.ALPHA

    if settings.COMPRESSION_METHOD == "AE" and not settings.REDUCED_SPACE:
        decoder = data.get("decoder")

        assert callable(decoder), "decoder must be a function if settings.COMPRESSION_METHOD=AE and bool(settings.REDUCED_SPACE) =False"

        V_w = decoder(w)
        V_w = V_w.flatten()

        Q = (G @ V_w - d)

    else:
        
        Q = (G @ V @ w - d)

    if sigma_2 and not R_inv:
        #When R is proportional to identity
        J_o = 0.5 / sigma_2 * np.dot(Q, Q)
    elif R_inv:
        J_o = 0.5 * Q.T @ R_inv @ Q
    else:
        raise ValueError("Either R_inv or sigma must be provided")

    J_b = 0.5 * alpha * np.dot(w, w)
    J = J_b + J_o

    if settings.DEBUG:
        print("J_b = {:.2f}, J_o = {:.2f}".format(J_b, J_o))
    return J


def grad_J(w, data, settings):
    device = data.get("device")
    d = data.get("d")
    G = data.get("G")
    V_trunc = data.get("V_trunc")
    V =  V_trunc if V_trunc is not None else data.get("V")
    V_grad = data.get("V_grad")
    R_inv = data.get("R_inv")

    sigma_2 = settings.OBS_VARIANCE
    alpha = settings.ALPHA

    if settings.COMPRESSION_METHOD == "AE" and not settings.REDUCED_SPACE:
        decoder = data.get("model").decode

        assert callable(V_grad), "V_grad must be a function if settings.COMPRESSION_METHOD=AE is used"
        model = data.get("model").to(device)

        w_tensor = torch.Tensor(w).to(device)
        V_w = decoder(w_tensor).detach().cpu().numpy()
        V_w = V_w.flatten()
        V_grad_w = V_grad(w_tensor).detach().cpu().numpy()

        Q = (G @ V_w - d)
        P = V_grad_w.T @ G.T
    else:
        Q = (G @ V @ w - d)
        P = V.T @ G.T

    if not R_inv and sigma_2:
        #When R is proportional to identity
        grad_o = (1.0 / sigma_2 ) * np.dot(P, Q)
    elif R_inv:
        J_o = 1.0 * P @ R_inv @ Q
    else:
        raise ValueError("Either R_inv or sigma must be non-zero")

    grad_J = alpha * w + grad_o

    return grad_J