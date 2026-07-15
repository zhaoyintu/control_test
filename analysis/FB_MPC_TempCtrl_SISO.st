FUNCTION_BLOCK FB_MPC_TempCtrl_SISO
(*
    SISO Hammerstein-FOPDT predictive controller, closed-form receding horizon (DMC-style).

    Proposed drop-in candidate for FB_AFPID_Base, scoped to the 100~400 degC band
    (cold start 0->100 degC is explicitly NOT a design target -- see accompanying notes).

    Design principle: this FB only executes FIXED, cheap per-scan arithmetic (a handful of
    multiply-adds, one monotonic table lookup, one clamp). Everything expensive -- system
    identification, the (S'WS+R)^-1 matrix solve, the observer gain design -- is done OFFLINE
    (Python/MATLAB) at commissioning time and loaded here as constants/parameters, exactly
    the way lrKP/lrKI/lrKD/lrIntegDetachMax/etc. are loaded into FB_AFPID_Base today.

    Run this FB at a COARSER, decoupled cycle (lrCycleT ~ 0.2-0.5 s) than the fast
    measurement/logging scan (~10 ms) -- see design note "why a slower MPC tick" below.

    NOT production code: the delay-FIFO / free-response indexing below is written for
    clarity and must be validated against the offline reference simulation (the same
    Python model used to fit K/tau/L and to build aDmcGain) before it is trusted on hardware.
*)
VAR_INPUT
    xEnable   : BOOL;
    xReset    : BOOL := FALSE;

    lrPV      : LREAL;    // measured PV, engineering units (degC)
    lrSV      : LREAL;    // setpoint, engineering units (degC)
    lrCycleT  : LREAL;    // MPC execution period Ts (s), e.g. 0.2-0.5

    // ---- identified Hammerstein-FOPDT model (fixed at commissioning) ----
    lrModelA    : LREAL;                          // discrete pole a = exp(-Ts/tau)
    nDeadSteps  : DINT;                           // round(L / Ts) -- expect 1-3 in this plant
    nMapPoints  : DINT;                           // breakpoints actually used below
    aMapU       : ARRAY[0..15] OF LREAL;          // MV% breakpoints, strictly ascending
    aMapV       : ARRAY[0..15] OF LREAL;          // steady-state model output at each breakpoint
                                                   // (the static nonlinear gain curve g(u))

    // ---- offline-computed DMC control-law gain row ----
    nHorizon    : DINT;                           // N used when aDmcGain was built (e.g. 60-150)
    aDmcGain    : ARRAY[0..199] OF LREAL;         // first row of (S'WS+R)^-1 S'W  [only 0..nHorizon-1 used]

    // ---- offset-free disturbance observer (steady-state Kalman gains, offline) ----
    lrObsKy     : LREAL;   // correction gain on model state estimate
    lrObsKd     : LREAL;   // correction gain on output-disturbance estimate

    // ---- constraints ----
    lrMVMin     : LREAL := 0.0;
    lrMVMax     : LREAL := 100.0;
    lrDMVMax    : LREAL := 0.0;    // max |change in MV| per MPC step, 0 = unconstrained
END_VAR
VAR_OUTPUT
    xBusy   : BOOL;
    xError  : BOOL;

    lrMV    : LREAL;   // controller output, 0-100 %
    lrYHat  : LREAL;   // model's own (pre-disturbance) predicted PV -- diagnostic
    lrDHat  : LREAL;   // estimated output disturbance / bias         -- diagnostic
    lrUSS   : LREAL;   // steady-state target MV = g^-1(SV - dHat)    -- diagnostic
END_VAR
VAR
    bFbState : E_FB_RUNST := E_FB_RUNST.unInit;

    aVDelay  : ARRAY[0..31] OF LREAL;   // FIFO of recently committed model-input v(k); [0]=newest
    aFree    : ARRAY[0..199] OF LREAL;  // free-response trajectory over the horizon

    lrYModel : LREAL;   // recursive model state (pre-disturbance correction)
    lrVLast  : LREAL;   // last committed v(k) = g(lrMV), held constant in the free-response projection
    lrPrevMV : LREAL;

    lrInnov  : LREAL;   // measurement innovation e(k) = PV - (yModel + dHat)
    lrVEff   : LREAL;
    lrYRun   : LREAL;
    lrDeltaV : LREAL;
    lrVCmd, lrUCmd : LREAL;

    i, k : DINT;
END_VAR

//Init MPC FB
IF xEnable THEN
    IF E_FB_RUNST.unInit = bFbState THEN
        FOR i := 0 TO 31 DO
            aVDelay[i] := 0.0;
        END_FOR
        lrYModel := lrPV;
        lrDHat   := 0.0;
        lrVLast  := 0.0;
        lrPrevMV := 0.0;
        bFbState := E_FB_RUNST.init;
    END_IF
ELSE
    bFbState := E_FB_RUNST.unInit;
    RETURN;
END_IF

IF xReset THEN
    bFbState := E_FB_RUNST.unInit;
    RETURN;
END_IF

IF E_FB_RUNST.run <> bFbState THEN
    xBusy := TRUE;
ELSE
    xBusy := FALSE;

    // === 1) offset-free observer: reconcile model with this scan's measurement ===
    // (lrYModel / lrDHat entering this line are last scan's predictions for *this* instant)
    lrInnov  := lrPV - (lrYModel + lrDHat);
    lrYModel := lrYModel + lrObsKy * lrInnov;
    lrDHat   := lrDHat   + lrObsKd * lrInnov;
    lrYHat   := lrYModel;

    // === 2) steady-state target: solve g(u_ss) = SV - dHat by inverting the static map ===
    lrUSS := FC_InterpInverse(aMapU, aMapV, nMapPoints, lrSV - lrDHat);

    // === 3) project the free response (v held at lrVLast, i.e. "if we do nothing more") ===
    // First nDeadSteps future samples are still fed by commands already sitting in the
    // delay FIFO (issued, not yet arrived); beyond that the FIFO has drained and every
    // sample sees lrVLast. Validate this indexing against the offline sim -- see file header.
    lrYRun := lrYModel;
    FOR k := 1 TO nHorizon DO
        IF k <= nDeadSteps THEN
            lrVEff := aVDelay[nDeadSteps - k];
        ELSE
            lrVEff := lrVLast;
        END_IF
        lrYRun     := lrModelA * lrYRun + (1.0 - lrModelA) * lrVEff;
        aFree[k-1] := lrYRun + lrDHat;   // add estimated bias -> offset-free prediction
    END_FOR

    // === 4) one dot-product: optimal first move, from the precomputed DMC gain row ===
    lrDeltaV := 0.0;
    FOR k := 0 TO nHorizon-1 DO
        lrDeltaV := lrDeltaV + aDmcGain[k] * (lrSV - aFree[k]);
    END_FOR
    lrVCmd := lrVLast + lrDeltaV;

    // === 5) invert static map back to MV%, apply constraints ===
    lrUCmd := FC_InterpInverse(aMapV, aMapU, nMapPoints, lrVCmd);  // note: V,U swapped -> inverse direction
    IF lrDMVMax > 0.0 THEN
        lrUCmd := LIMIT(lrPrevMV - lrDMVMax, lrUCmd, lrPrevMV + lrDMVMax);
    END_IF
    lrMV := LIMIT(lrMVMin, lrUCmd, lrMVMax);

    // === 6) commit: re-derive v actually applied (post-clamp!), advance FIFO and model state ===
    lrVLast := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMV);
    FOR i := 31 TO 1 BY -1 DO
        aVDelay[i] := aVDelay[i-1];
    END_FOR
    aVDelay[0] := lrVLast;
    lrPrevMV   := lrMV;
END_IF

IF E_FB_RUNST.init = bFbState THEN
    bFbState := E_FB_RUNST.run;
END_IF

(*
  FC_InterpLookup(aX, aY, n, x)   : monotonic breakpoint lookup, linear interpolation, y = f(x)
  FC_InterpInverse(aX, aY, n, y)  : same table, solved for x = f^-1(y) (aX,aY both ascending
                                    since g(u) is monotonic increasing -- confirm this holds
                                    all the way down to MV~0 before relying on it there)
  Both are ~15-line binary-search + lerp helpers, not shown here.
*)
