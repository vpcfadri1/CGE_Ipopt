import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
import pprint
import calibrate


# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "PH_SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)
# declare sets of variables
u = (
    "AGR", "MIN", "MAN", "ESW", "CON", "WRT", "TRS", "AFS", "INF", "FIN", 
    "REA", "PBS", "PAD", "EDU", "HHS", "OTH", "CAP", "LAB", "IDT", "TRF", 
    "HOH", "GOV", "INV", "EXT"
)

ind = (
    "AGR", "MIN", "MAN", "ESW", "CON", "WRT", "TRS", "AFS", "INF", "FIN", 
    "REA", "PBS", "PAD", "EDU", "HHS", "OTH"
)

h = ("CAP", "LAB")


def check_square():
    """
    this function tests whether the SAM is a square matrix.
    """
    
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    sam_small.to_numpy(dtype=None, copy=True)

    if not sam_small.shape[0] == sam_small.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[0]} columns"
        )


def row_total():
    """
    This function tests whether the row sums
    of the SAM equal the expected value.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    row_sum = sam_small.sum(axis=0)
    row_sum = pd.Series(row_sum)
    return row_sum


def col_total():
    """
    This function tests whether column sums
    of the SAM equal the expected values.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    col_sum = sam_small.sum(axis=1)
    col_sum = pd.Series(col_sum)
    return col_sum


def row_col_equal():
    """
    This function tests whether row sums
    and column sums of the SAM are equal.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    np.testing.assert_allclose(row_sum, col_sum)




def iterative_runner():
    """
    This function solves the CGE model using the iterative model
    """

    # iterative model parameters
    dist = 10
    tpi_tol = 1e-10
    tpi_iter = 0
    tpi_max_iter = 5
    xi = 0.1
    tiny = 0.000001
    damping = 0.1

    # Initialize all Variables
    pfbar = np.ones(len(h)) 
    pqbar = np.ones(len(ind))
    pdbar = np.ones(len(ind))
    er = 1

    # Load data and parameters classes
    d = calibrate.model_data(sam, h, ind) # correct
    p = calibrate.parameters(d, ind, sam) # correct
    
    # numeraire
    py = np.ones(len(ind))

    # Initialize Variables
    Zbar = d.Z0 
    Ffbar = d.Ff0
    Qbar = d.Q0
    Mbar = d.M0
    Ebar = d.E0
    Dbar = d.D0
    Ybar = d.Y0

    residuals = {
        'factor': np.zeros(len(h)),
        'goods': np.zeros(len(ind)),
        'domestic': np.zeros(len(ind)),
        'trade': 0.0
    }

    while dist > tpi_tol and tpi_iter < tpi_max_iter:
        # Step 1: Compute for Intermediate Variables
        pe = firms.eqpe(er, d.pWe)
        pm = firms.eqpm(er, d.pWm)
        Td = gov.eqTd(p.taud, pfbar, Ffbar)
        Sp = agg.eqSp(p.ssp, pfbar, Ffbar)
        I = hh.eqI(pfbar, Ffbar, Sp, Td)
        Xp = hh.eqXp(p.alpha, I, pqbar) 
        F = hh.eqF(p.beta, py, Ybar, pfbar)
        X = firms.eqX(p.ax, Zbar)
        pz = firms.eqpz(p.ay, p.ax, py, pqbar)
        Tm = gov.eqTm(p.taum, pm, Mbar) #6.8
        Tz = gov.eqTz(p.tauz, pz, Zbar) #6.7 
        Sg = gov.eqSg(p.ssg, Td, Tz, Tm) #6.12
        Xg = gov.eqXg(p.mu, d.XXg0) #6.9
        Xv = firms.eqXv(p.lam, Sp, d.Sf0, er, Sg, pqbar)
        Q = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, Mbar, Dbar)
        M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pqbar, pm, p.taum) #6.18
        D_619 = firms.eqD(p.gamma, p.deltad, p.eta, Qbar, pqbar, pdbar) #6.19
        E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
        Z = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, Dbar)#6.20
        Y = firms.eqY(p.ay, Zbar) #6.4


        #Step 2: Check Equations of Relations
        print("F (Factor Demand):", F)
        print("Ffbar (Factor Supply):", Ffbar)
        residuals['factor'] = np.array(agg.eqpf(F, Ffbar))
        print("Factor Residuals (demand - supply):", residuals['factor'])

        print("Q (Composite Supply):", Q)
        print("Total Demand (Xp + Xg + Xv + X):", Xp + Xg + Xv + np.sum(X, axis=1))
        residuals['goods'] = np.array(agg.eqpqerror(Q, Xp, Xg, Xv, X))


        print("D_619 (Domestic Demand):", D_619)
        print("Dbar (Domestic Supply):", Dbar)
        residuals['domestic'] = np.array(D_619 - Dbar)


        residuals['trade'] = float(agg.eqbop(d.pWe, d.pWm, E, M, d.Sf0).sum())

        #Step 3: Compute Distance Metric
        dist = max(
            np.max(np.abs(residuals['factor'])),
            np.max(np.abs(residuals['goods'])),
            np.max(np.abs(residuals['domestic'])),
            np.abs(residuals['trade'])
        )
        
        
        #Step 4: Adjust Undefined Variables
        if dist > tpi_tol:
            # Factor prices: decrease if demand > supply, increase if demand < supply
            pfbar += damping * tiny * np.sign(residuals['factor'])
            pfbar = np.maximum(pfbar, 1e-10)  # Ensure positive prices

            # Composite goods prices: decrease if supply > demand, increase if supply < demand
            pqbar -= damping * tiny * np.sign(residuals['goods'])
            pqbar = np.maximum(pqbar, 1e-10)

            # Domestic prices: decrease if D_619 > D_622, increase if D_619 < D_622
            pdbar += damping * tiny * np.sign(residuals['domestic'])
            pdbar = np.maximum(pdbar, 1e-10)

            # Exchange rate: decrease if exports + S^f > imports, increase if exports + S^f < imports
            er += damping * tiny * np.sign(residuals['trade'])
            er = max(er, 1e-10)
        
        #Step 5: Update Endogenous Variables
        Zbar = Z  # Activity levels
        Qbar = Q  # Composite goods supply
        Mbar = M  # Imports
        Ebar = E  # Exports
        Dbar = D_619  # Domestic goods demand
        Ybar = Y
        tpi_iter = tpi_iter + 1
        
        print(f"Iteration {tpi_iter}: Distance = {dist}")
        print(f"Factor Residuals: {residuals['factor']}")
        print(f"Goods Residuals: {residuals['goods']}")
        print(f"Domestic Residuals: {residuals['domestic']}")
        print(f"Trade Residual: {residuals['trade']}")
        print(f"Distance: {dist}")
        print(f"pfbar: {pfbar}")
        print(f"pqbar: {pqbar}")
        print(f"pdbar: {pdbar}")
        print(f"er: {er}")
        

    if dist <= tpi_tol:
        print(f"Converged after {tpi_iter} iterations. Distance = {dist}")
    else:
        print(f"Failed to converge after {tpi_max_iter} iterations. Distance = {dist}")

    results = {
        'pf': pfbar,
        'pq': pqbar,
        'pd': pdbar,
        'er': er,
        'residuals': residuals,
        'Y': Y,
        'Z': Z,
        'F': F,
        'X': X,
        'pz': pz,
        'Td': Td,
        'Tz': Tz,
        'Tm': Tm,
        'Sp': Sp,
        'Sg': Sg,
        'Xp': Xp,
        'Xg': Xg,
        'Xv': Xv,
        'Q': Q,
        'M': M,
        'D': D_619,
        'E': E,
        'pe': pe,
        'pm': pm
    }
    pprint.pprint(results)