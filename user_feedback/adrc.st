FUNCTION_BLOCK FB_ADRC_Base
VAR_INPUT
    xEnable    : BOOL;
    xReset     : BOOL := FALSE;

    lrPV       : LREAL;    // measured PV, engineering units (degC)
    lrSV       : LREAL;    // setpoint, engineering units (degC)
    lrCycleT   : LREAL;    // execution period Ts (s)

	lrK        : LREAL;
    lrTau      : LREAL;    // dominant time constant (s) -- b0 = 1/lrTau, nothing else to guess
    lrWc       : LREAL;    // controller bandwidth (rad/s) -- sets closed-loop speed
    lrWo       : LREAL;    // observer bandwidth (rad/s), typ. 3-10x lrWc
    lrWr       : LREAL;    // reference-shaping bandwidth (rad/s)

    nDeadSteps : DINT;     // round(L / Ts) -- dead-time compensation for the ESO
    nMapPoints : DINT;     // breakpoints actually used in aMapU/aMapV
    aMapU      : ARRAY[0..15] OF LREAL;   // MV% breakpoints, strictly ascending
    aMapV      : ARRAY[0..15] OF LREAL;   // steady-state model output g(u) at each breakpoint
                                          // (from real long-hold pulse tests, not guessed)
										  
	xSvChanged : BOOL;

    lrMVMin    : LREAL := 0.0;
    lrMVMax    : LREAL := 100.0;
END_VAR
VAR_OUTPUT
    xBusy  : BOOL;
    xError : BOOL;

    lrMV   : LREAL;   // controller output, 0-100 %
    lrZ1   : LREAL;   // ESO state estimate (~PV)               -- diagnostic
    lrZ2   : LREAL;   // ESO total-disturbance estimate         -- diagnostic
    lrRV1  : LREAL;   // shaped reference trajectory            -- diagnostic
END_VAR
VAR
    bFbState : E_FB_RUNST := E_FB_RUNST.unInit;

    aVDelay    : ARRAY[0..31] OF LREAL;   // FIFO of recent v=g(u) commands; [0] = newest
    lrB0       : LREAL;                   // = 1/lrTau, computed once at init
    lrRV2      : LREAL;                   // shaped reference's own rate (internal to TD)
    lrRV1Old   : LREAL;
    lrErrO     : LREAL;
    lrU0       : LREAL;
    lrVCmd     : LREAL;
    lrUCmd     : LREAL;
    lrVApplied : LREAL;
    lrVDelayed : LREAL;
	
    i : DINT;
END_VAR

//Init ADRC FB
IF xEnable THEN
    IF E_FB_RUNST.unInit = bFbState THEN
        //lrB0 := 1.0 / lrTau;
		lrB0 := lrK;
        FOR i := 0 TO 31 DO
            //aVDelay[i] := FC_InterpLookup(aMapU, aMapV, nMapPoints, 0.0);
			aVDelay[i] := 0.0;
        END_FOR
        lrZ1  := lrPV;
        lrZ2  := 0.0;
        lrRV1 := lrSV;
        lrRV2 := 0.0;
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
	
	lrB0 := lrK;

    // === 1) reference shaping (critically damped 2nd order tracking filter) ===
    // Decouples "how fast we approach a new SV" from "how fast the loop rejects
    // disturbance". Tune lrWr against an explicit overshoot budget.
    lrRV1Old := lrRV1;
    lrRV1 := lrRV1 + lrCycleT * lrRV2;
    lrRV2 := lrRV2 + lrCycleT * (-2.0*lrWr*lrRV2 - lrWr*lrWr*(lrRV1Old - lrSV));

    // === 2) ESO: reconcile estimate with this scan's measurement ===
    // Uses the delayed v (nDeadSteps old), not the just-committed one -- this is what
    // "vanilla ADRC applied to a plant with known dead time" should actually do.
    lrVDelayed := aVDelay[nDeadSteps];
    lrErrO := lrPV - lrZ1;
    lrZ1 := lrZ1 + lrCycleT * (lrZ2 + lrB0*lrVDelayed + 2.0*lrWo*lrErrO);
    lrZ2 := lrZ2 + lrCycleT * (lrWo*lrWo*lrErrO);

    // === 3) control law: proportional feedback on the ESO's clean estimate,
    //        cancel the estimated total disturbance, invert the KNOWN static map ===
    lrU0   := lrWc * (lrRV1 - lrZ1);
    lrVCmd := (lrU0 - lrZ2) / lrB0;                                   // temperature-equivalent demand
    //lrUCmd := FC_InterpInverse(aMapU, aMapV, nMapPoints, lrVCmd);     // -> real MV%
	lrUCmd := lrVCmd;
    lrMV   := LIMIT(lrMVMin, lrUCmd, lrMVMax);

    // === 4) advance the delay FIFO with what v ACTUALLY happened (post-clamp!) ===
    //lrVApplied := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMV);
    FOR i := 31 TO 1 BY -1 DO
        aVDelay[i] := aVDelay[i-1];
    END_FOR
    //aVDelay[0] := lrVApplied;
	aVDelay[0] := lrMV;
	
END_IF

IF E_FB_RUNST.init = bFbState THEN
    bFbState := E_FB_RUNST.run;
END_IF

(*
  FC_InterpLookup(aX, aY, n, x)   : monotonic breakpoint lookup, linear interpolation, y = g(x)
  FC_InterpInverse(aX, aY, n, y)  : same table, solved for x = g^-1(y) (aX,aY both ascending
                                    since g(u) is monotonic increasing)
  Both are ~15-line binary-search + lerp helpers -- same two helpers already needed for the
  MPC design (FB_MPC_TempCtrl_SISO.st), reusable as-is.
*)
