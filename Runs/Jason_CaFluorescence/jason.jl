u0 = [-80.0, 1.0, 0.0, 0.322, 0.7]
p = (450.0, 50.0, 2.0, 3.0, 10.0, 2.0)

function dynamics(du, u, p, t)
    ### Consts
    tauNbar,tauH, taur0, taur1, thrT = 10.0, 1.0, 40.0, 17.5, 68.0

    ENa, EK, EL, ECa = 45.0, -90.0, -70.0, 85.0

    thetaM, thetaN, thetaH, sigmaM, sigmaN, sigmaH, thetaS, sigmaS = -35.0,-30.0, -37.4, -5.0,-5.0, 4.3,-30.0,-12.0

    thetaaT ,sigmaaT ,thetabT, sigmabT, thetarT, sigmarT = -65.0,-7.8,.4, -.1,-68.0,2.2

    sgmrt, phirT, thetaRF, sigmaRF, thetaRS = 2.2, 1.0, -105.0, 5.0, -105.0

    sigmaRS, kr, ks, Cm = 25.0,.35, .35, 100.0

    f, eps, kca = .1, .0015, .3
    ###

    v,h,n,ca,rT = u
    gNa,gK,gSK,gCaT, gCaL, gL = p

    sig(x, y, z) = 0.5*(1.0 + tanh(-(x-y)/(2.0*z)))
    E(x, y, z) = exp(-(x-y)/z)

    # Na+ and K+ Equations and Currents
    minf = sig(v, thetaM, sigmaM)
    ninf = sig(v, thetaN, sigmaN)
    hinf = sig(v, thetaH, sigmaH)

    tauN = tauNbar/cosh((v-thetaN)/(2*sigmaN))

    iNa = gNa*(minf^3)*h*(v-ENa)
    iK = gK*(n^4)*(v-EK)

    # L-Type Ca++ Equations and Current
    sinf = sig(v, thetaS, sigmaS)
    iCaL = gCaL*(sinf^2)*(v - ECa)

    # T-Type Ca++ Equations and Current
    aTinf = sig(v, thetaaT, sigmaaT)
    bTinf = sig(rT, thetabT, sigmabT)-sig(0, thetabT, sigmabT)
    iCaT=gCaT*(aTinf^3)*(bTinf^3)*(v - ECa)

    rTinf = sig(v, thetarT, sigmarT)
    taurT = taur0+taur1*sig(v, thrT, sgmrt)

    # SK Equations and Current
    kinf = (ca^4)/(ca^4+ks^4)
    iSK = gSK*kinf*(v-EK)

    # Leak current
    iL = gL*(v-EL)

    #v,h,n,rf,rs,ca,rT
    du[1] = (-iNa-iK-iCaL-iCaT-iSK-iL+stim_func(t))/Cm
    du[2] = (hinf-h)/tauH
    du[3] = (ninf-n)/tauN
    du[4] = -f*(eps*(iCaL+iCaT)+ kca*(ca-0.1))
    du[5] = phirT*(rTinf-rT)/taurT
 end