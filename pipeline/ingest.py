#!/usr/bin/python3
"""Ingests data and creates """
import numpy as np
from helpers import VarDataAssimilationPipeline as VarDa
import settings


def main():
    vda = VarDa()

    field_name = "Tracer"
    fps_sorted = vda.get_sorted_fps_U(settings.DATA_FP, field_name)
    print("Got sorted list")

    X = vda.create_X_from_fps(fps_sorted, field_name)
    print(X.shape)

    np.save(settings.X_FP, X)

    V = vda.create_V_from_X(X)
    V_T = vda.trunc_SVD(V)

if __name__ == "__main__":
    main()
