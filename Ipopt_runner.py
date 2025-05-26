from pyomo.environ import *
import numpy as np
import pandas as pd
import os
import calibrate
import shocker
import pandas as pd
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.param import _ParamData
import ast
import visualize_data

current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "PH_SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)

# declare sets of variables
u = (
    "AFF", "MAQ", "MFG", "ESWW", "CNS", "TRD", "TAS", "AFSA", "IAC", "FIA", 
    "REOD", "PBS", "PAD", "EDUC", "HHSW", "OS", "CAP", "LAB", "IDT", "TRF", 
    "HOH", "GOV", "INV", "EXT"
)

ind = (
    "AFF", "MAQ", "MFG", "ESWW", "CNS", "TRD", "TAS", "AFSA", "IAC", "FIA", 
    "REOD", "PBS", "PAD", "EDUC", "HHSW", "OS",
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

def runner(return_period: int, affected_regions: list):

    # Get new percentage values
    sector_shocks = shocker.get_shocks(return_period, affected_regions)
    print("Sector Shocks:", sector_shocks)

    # Create model
    model = ConcreteModel()

    # Define Sets
    model.ind = Set(initialize=["AFF", "MAQ", "MFG", "ESWW", "CNS", "TRD", "TAS", "AFSA", 
                                "IAC", "FIA", "REOD", "PBS", "PAD", "EDUC", "HHSW", "OS"])
    model.h = Set(initialize=['CAP', 'LAB'])

    # sector_shocks = {
    #     'AFF': -0.1,
    #     'MAQ': -0.2,
    #     'MFG': -0.3,
    #     'ESWW': -0.1,
    #     'CNS': -0.2,
    #     'TRD': -0.3,
    #     'TAS': -0.1,
    #     'AFSA': -0.2,
    #     'IAC': -0.3,
    #     'FIA': -0.1,
    #     'REOD': -0.2,
    #     'PBS': -0.3,
    #     'PAD': -0.1,
    #     'EDUC': -0.2,
    #     'HHSW': -0.3,
    #     'OS': -0.1,
    # }
    # Define Parameters
    d = calibrate.model_data(sam, h, ind)
    p = calibrate.parameters(d, ind, sam, shocks=sector_shocks) 

    # Applying shock to b_j = scaling coefficient in the composite factor production function
    # for j, shock in sector_shocks.items():
    #     if j in p.b:
    #         p.b[j] *= (1 + shock)

    # Add to Pyomo params
    model.tauz = Param(model.ind, initialize=p.tauz.to_dict(), within=Reals)
    model.taum = Param(model.ind, initialize=p.taum.to_dict(), within=Reals) 
    model.FF = Param(model.h, initialize=d.Ff0.to_dict(), within=Reals)
    model.Sf = Param(initialize=p.Sf0.loc['INV', 'EXT'], within=Reals)
    model.pWe = Param(initialize=1, within=Reals)
    model.pWm = Param(initialize=1, within=Reals)
    

    model.sigma = Param(model.ind, initialize=p.sigma.to_dict(), within=Reals)
    model.psi = Param(model.ind, initialize=p.psi.to_dict(), within=Reals)
    model.eta = Param(model.ind, initialize=p.eta.to_dict(), within=Reals)
    model.phi = Param(model.ind, initialize=p.phi.to_dict(), within=Reals)

    model.alpha = Param(model.ind, initialize=p.alpha.to_dict(), within=Reals)
    model.beta = Param(model.h, model.ind, initialize={(h,j): p.beta.loc[h,j] for h in model.h for j in model.ind}, within=Reals)
    model.b = Param(model.ind, initialize=p.b.to_dict(), within=Reals)
    model.ax = Param(model.ind, model.ind, initialize={(i,j): p.ax.loc[i,j] for i in model.ind for j in model.ind}, within=Reals)
    model.ay = Param(model.ind, initialize=p.ay.to_dict(), within=Reals)
    model.mu = Param(model.ind, initialize=p.mu.to_dict(), within=Reals)
    model.lam = Param(model.ind, initialize=p.lam.to_dict(), within=Reals)
    model.deltam = Param(model.ind, initialize=p.deltam.to_dict(), within=Reals)
    model.deltad = Param(model.ind, initialize=p.deltad.to_dict(), within=Reals)
    model.gamma = Param(model.ind, initialize=p.gamma.to_dict(), within=Reals)
    model.xie = Param(model.ind, initialize=p.xie.to_dict(), within=Reals)
    model.xid = Param(model.ind, initialize=p.xid.to_dict(), within=Reals)
    model.theta = Param(model.ind, initialize=p.theta.to_dict(), within=Reals)
    model.ssp = Param(initialize=float(p.ssp), within=Reals)
    model.ssg = Param(initialize=float(p.ssg), within=Reals)
    model.tau_d = Param(initialize=float(p.taud), within=Reals)
    
    # Add to Pyomo Variables
    model_variables = ["Y", "F", "X", "Z", "Xp", "Xg", "Xv", "E", "M", "Q", "D",
    "pf", "py", "pz", "pq", "pe", "pm", "pd", 
    "epsilon", "Sp", "Sg", "Td", "Tz", "Tm"]
    model.Y = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Y0.to_dict())    
    model.F = Var(model.h, model.ind, bounds=(0.0001, None),  within=NonNegativeReals, initialize={(h,j): d.F0.loc[h,j] for h in model.h for j in model.ind})
    model.X = Var(model.ind, model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize={(i,j): d.X0.loc[i,j] for i in model.ind for j in model.ind})
    model.Z = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Z0.to_dict())
    model.Xp = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Xp0['HOH'].to_dict())
    model.Xg = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Xg0['GOV'].to_dict())
    model.Xv = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Xv0['INV'].to_dict())
    model.E = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.E0.to_dict())
    model.M = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.M0.to_dict())
    model.Q = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.Q0.to_dict())
    model.D = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=d.D0.to_dict())
    model.pf = Var(model.h, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.py = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.pz = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.pq = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.pe = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.pm = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.pd = Var(model.ind, bounds=(0.0001, None), within=NonNegativeReals, initialize=1)
    model.epsilon = Var(initialize=1, within=Reals)
    model.Sp = Var(bounds=(0.0001, None),within=NonNegativeReals, initialize=p.Sp0.loc['INV', 'HOH'])
    model.Sg = Var(bounds=(0.0001, None),within=NonNegativeReals, initialize=p.Sg0.loc['INV', 'GOV'])
    model.Td = Var(bounds=(0.0001, None),within=NonNegativeReals, initialize=d.Td0.loc['GOV', 'HOH'])
    model.Tz = Var(model.ind, bounds=(0.0000, None), within=NonNegativeReals, initialize=d.Tz0.loc['IDT'].to_dict())
    model.Tm = Var(model.ind, bounds=(0.0000, None), within=NonNegativeReals, initialize=d.Tm0.loc['TRF'].to_dict())
    
    visualize_data.export_pyomo_variables_to_excel(model, model_variables, filename="cge_variables.xlsx")
    # Display paramaters
    # model.Q.display()
    # model.pf.display()
    # model.py.display()
    # model.pz.display()
    # model.pq.display()
    # model.F.display()
    # model.Y.display()
    var = getattr(model,"epsilon")
    
    data = {k: v.value for k, v in var.items() if isinstance(v, _GeneralVarData)}
    print(data)
    # Define Equations

    # Domestic Production:
    def eq_6_1(model, j):
        return model.Y[j] == model.b[j] * prod(model.F[h,j]**model.beta[h,j] for h in model.h)
    model.eq_6_1 = Constraint(model.ind, rule=eq_6_1)

    def eq_6_2(model, h, j):
        return model.F[h,j] == model.beta[h,j] * model.Y[j] * model.py[j] / model.pf[h]
    model.eq_6_2 = Constraint(model.h, model.ind, rule=eq_6_2)
    
    def eq_6_3(model, i, j):
        return model.X[i,j] == model.ax[i,j] * model.Z[j]
    model.eq_6_3 = Constraint(model.ind, model.ind, rule=eq_6_3)

    def eq_6_4(model, j):
        return model.Y[j] == model.ay[j] * model.Z[j]
    model.eq_6_4 = Constraint(model.ind, rule=eq_6_4)

    def eq_6_5(model, j):
        return model.pz[j] == model.ay[j] * model.py[j] + sum(model.ax[i,j] * model.pq[i] for i in model.ind)
    model.eq_6_5 = Constraint(model.ind, rule=eq_6_5)
    
    # Government:
    def eq_6_6(model):
        return model.Td == model.tau_d * sum(model.pf[h] * model.FF[h] for h in model.h)
    model.eq_6_6 = Constraint(rule=eq_6_6)

    def eq_6_7(model, j):
        return model.Tz[j] == model.tauz[j] * model.pz[j] * model.Z[j]
    model.eq_6_7 = Constraint(model.ind, rule=eq_6_7)

    def eq_6_8(model, i):
        return model.Tm[i] == model.taum[i] * model.pm[i] * model.M[i]
    model.eq_6_8 = Constraint(model.ind, rule=eq_6_8)

    def eq_6_9(model, i):
        return model.Xg[i] == (model.mu[i] / model.pq[i]) * (
        model.Td + sum(model.Tz[j] for j in model.ind) + sum(model.Tm[j] for j in model.ind) - model.Sg)
    model.eq_6_9 = Constraint(model.ind, rule=eq_6_9)

    # Investment and Savings:
    def eq_6_10(model, i):
        return model.Xv[i] == (model.lam[i] / model.pq[i]) * (model.Sp + model.Sg + model.epsilon * model.Sf)
    model.eq_6_10 = Constraint(model.ind, rule=eq_6_10)

    def eq_6_11(model):
        return model.Sp == model.ssp * sum(model.pf[h] * model.FF[h] for h in model.h)
    model.eq_6_11 = Constraint(rule=eq_6_11)

    def eq_6_12(model):
        return model.Sg == model.ssg * (
            model.Td + sum(model.Tz[j] for j in model.ind) + sum(model.Tm[j] for j in model.ind))
    model.eq_6_12 = Constraint(rule=eq_6_12)

    # Household:
    def eq_6_13(model, i):
        return model.Xp[i] == (model.alpha[i] / model.pq[i]) * (
            sum(model.pf[h] * model.FF[h] for h in model.h) - model.Sp - model.Td)
    model.eq_6_13 = Constraint(model.ind, rule=eq_6_13)

    # Export and import prices and the BOP constraint:
    def eq_6_14(model, i):
        return model.pe[i] == model.epsilon * model.pWe
    model.eq_6_14 = Constraint(model.ind, rule=eq_6_14)

    def eq_6_15(model, i):
        return model.pm[i] == model.epsilon * model.pWm
    model.eq_6_15 = Constraint(model.ind, rule=eq_6_15)
    
    # def eq_6_16(model):
    #     return sum(model.pWe * model.E[i] for i in model.ind) + model.Sf == sum(model.pWm * model.M[i] for i in model.ind)
    # model.eq_6_16 = Constraint(rule=eq_6_16)

    # Substitution between imports and domestic goods:
    # (Armington Composite)

    def eq_6_17(model, i):
        return model.Q[i] == model.gamma[i] * (
            model.deltam[i] * model.M[i]**model.eta[i] + model.deltad[i] * model.D[i]**model.eta[i])**(1/model.eta[i])
    model.eq_6_17 = Constraint(model.ind, rule=eq_6_17)

    def eq_6_18(model, i):
        return model.M[i] == (
            (model.gamma[i]**model.eta[i] * model.deltam[i] * model.pq[i]) /
            ((1 + model.taum[i]) * model.pm[i]))**(1/(1 - model.eta[i])) * model.Q[i]
    model.eq_6_18 = Constraint(model.ind, rule=eq_6_18)
    
    def eq_6_19(model, i):
        return model.D[i] == (
            (model.gamma[i]**model.eta[i] * model.deltad[i] * model.pq[i]) /
            model.pd[i])**(1/(1 - model.eta[i])) * model.Q[i]
    model.eq_6_19 = Constraint(model.ind, rule=eq_6_19)

    # Transformation between exports and domestic goods:
    def eq_6_20(model, i):
        return model.Z[i] == model.theta[i] * (
            model.xie[i] * model.E[i]**model.phi[i] + model.xid[i] * model.D[i]**model.phi[i])**(1/model.phi[i])
    model.eq_6_20 = Constraint(model.ind, rule=eq_6_20)

    def eq_6_21(model, i):
        return model.E[i] == (
            (model.theta[i]**model.phi[i] * model.xie[i] * (1 + model.tauz[i]) * model.pz[i]) /
            model.pe[i])**(1/(1 - model.phi[i])) * model.Z[i]
    model.eq_6_21 = Constraint(model.ind, rule=eq_6_21)

    def eq_6_22(model, i):
        return model.D[i] == (
            (model.theta[i]**model.phi[i] * model.xid[i] * (1 + model.tauz[i]) * model.pz[i]) /
            model.pd[i])**(1/(1 - model.phi[i])) * model.Z[i]
    model.eq_6_22 = Constraint(model.ind, rule=eq_6_22)

    # Market-clearing conditions:
    def eq_6_23(model, i):
        return model.Q[i] == model.Xp[i] + model.Xg[i] + model.Xv[i] + sum(model.X[i,j] for j in model.ind)
    model.eq_6_23 = Constraint(model.ind, rule=eq_6_23)

    def eq_6_24(model, h):
        return sum(model.F[h,j] for j in model.ind) == model.FF[h]
    model.eq_6_24 = Constraint(model.h, rule=eq_6_24)
    
    def objective_rule(model):
        return prod(model.Xp[i] ** model.alpha[i] for i in model.ind)
    model.UU = Objective(rule=objective_rule, sense=maximize)
    
    # Fix numeraire
    model.epsilon.fix(1)
    
    solver = SolverFactory('ipopt')

    # Set IPOPT options
    # solver.options['max_iter'] = 5000           # maximum number of iterations (default was 3000)
    # solver.options['tol'] = 1e-6                # relative tolerance (default was 1e-8)
    # solver.options['constr_viol_tol'] = 1e-6    # constraint violation tolerance (default was 1e-8)
    # solver.options['acceptable_tol'] = 1e-4     # acceptable solution tolerance
    # solver.options['print_level'] = 5
    # solver.options['output_file'] = "ipopt_log.txt"

    results = solver.solve(model, tee=True)
    visualize_data.shocked_variables_to_excel(model, model_variables, "cge_variables.xlsx", "shocked_variables.xlsx")
    
    # Generate charts
    visualize_data.create_charts("shocked_variables.xlsx", "Z", "GDP per Sector, Z", two_vars=True)
    visualize_data.create_charts("shocked_variables.xlsx", "pq", "Final Good Price, pq")

return_period = 50
affected_regions = ["NCR", #"CAR", "II", "V", "VIII", "X"]
                    "CAR", 
                    "I", 
                    "II", 
                    "III", 
                    "IVA", 
                    "IVB", 
                    "V", 
                    "VI", 
                    "VII", 
                    "VIII", 
                    "IX", 
                    "X", 
                    "XI", 
                    "XII", 
                    "XIII", 
                    "BARMM"] 
runner(return_period, affected_regions)