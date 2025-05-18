def runner():
    """
    This function solves the CGE model
    """

    # solve cge_system
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1

    # pvec = pvec_init; initialize all industry and factor endowment prices
    pvec = np.ones(len(ind) + len(h)) 

    # Load data and parameters classes
    d = calibrate.model_data(sam, h, ind) # correct
    p = calibrate.parameters(d, ind, sam) # correct
    
    # R = d.R0
    er = 1 # numeraire

    Zbar = d.Z0 
    Ffbar = d.Ff0
    # Kdbar = d.Kd0; N
    Qbar = d.Q0
    pdbar = pvec[0 : len(ind)]

    pm = firms.eqpm(er, d.pWm)

    while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
        tpi_iter += 1
        cge_args = [p, d, ind, h, Zbar]

        results = opt.root(cge.cge_system, pvec, args=cge_args, method="lm", tol=1e-5)
        pprime = results.x

        # Updated Prices
        pyprime = pprime[0 : len(ind)]
        pfprime = pprime[len(ind) : len(ind) + len(h)]
        pyprime = Series(pyprime, index=list(ind))
        pfprime = Series(pfprime, index=list(h))
        
        pvec = pprime

        # Nobuhiro Equations (not including 6.1, 6.17, 6.18, 6.20 6.21, 6.22, 6.24)
        pe = firms.eqpe(er, d.pWe) #6.14
        pm = firms.eqpm(er, d.pWm) #6.15
        pq = firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)  #6.23
        pz = firms.eqpz(p.ay, p.ax, pyprime, pq) #6.5
        Td = gov.eqTd(p.taud, pfprime, Ffbar) #6.6
        F = hh.eqF(p.beta, pyprime, d.Y0, pfprime) #6.2         
        Xg = gov.eqXg(p.mu, d.XXg0) #6.9
        Sp = agg.eqSp(p.ssp, pfprime, Ffbar) #6.11
        I = hh.eqI(pfprime, Ffbar, Sp, Td)
        D = firms.eqD(p.gamma, p.deltad, p.eta, Qbar, pq, pdbar) #6.19
        E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar) #6.21
        D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar) #6.22
        M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum) #6.18
        Z = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)#6.20
        Zprime = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D) #6.20
      
        Xp = hh.eqXp(p.alpha, I, pq) #6.13
        Tm = gov.eqTm(p.taum, pm, M) #6.8
        Tz = gov.eqTz(p.tauz, pz, Z) #6.7
        Sg = gov.eqSg(p.ssg, Td, Tz, Tm) #6.12
       
        Xv = firms.eqXv(p.lam, Sp, d.Sf0, er, Sg, pq) #6.10
        Y = firms.eqY(p.ay, Zbar) #6.4

        Qprime = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D) #6.17
        pdprime = firms.eqpd(p.gamma, p.deltam, p.eta, Qprime, pq, D) #Unknown

      
        Ffprime = d.Ff0
        dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()
        #print("Distance at iteration ", tpi_iter, " is ", dist)

        # update arguments
    
        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar 
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        
        Q = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)
        #print("Model solved, Qbar = ", Qbar.to_markdown())
        #print("Model solved, Qprime = ", Qprime.to_markdown())
        

    #print("Model solved, Q = ", Q.to_markdown())
    #print("Core Equations Error")
    #print(cge.cge_system(pprime, cge_args))
    #Balance of Payments Error


    return Q
# Removed Equations
# Kk = agg.eqKk(pfprime, Ffbar, R, p.lam, pq)
# Trf = gov.eqTrf(p.tautr, pfprime, Ffbar)
# Kf = agg.eqKf(Kk, Kdbar)
# Fsh = firms.eqFsh(R, Kf, er)
# Ffprime["CAP"] = R * Kk * (p.lam * pq).sum() / pfprime.iloc[1]
# Kdbar = xi * Kdprime + (1 - xi) * Kdbar
# Kdprime = agg.eqKd(d.g, Sp, p.lam, pq)

