FUNCTION_BLOCK FB_AFPID_Base
VAR_INPUT
	xEnable : BOOL;
	xReset  : BOOL := FALSE;
	
	lrPV  : LREAL;   // 0~100 : 0~1000℃	   
	lrPVr : LREAL;   // refPV value

	lrSV  : LREAL;	
	lrKP : LREAL;
	lrKI : LREAL;
	lrKD : LREAL;
	
	lrCycleT : LREAL;
	
	xSvChanged : BOOL;
	xSvChangeClrInteg : BOOL;
	
	xEnIntegClrSet : BOOL;
	lrIntegClrSetV : LREAL;
	xEnIntegUpdate : BOOL;
	lrIntegUpdate  : LREAL;
	
	xEnIntegDetach   : BOOL;
	xForceIntegDetach : BOOL;
	
	lrIntegDetachMax : LREAL;
	lrIntegDetachMin : LREAL;
	xEnIntegAntiWind : BOOL;
	lrIntegAntiWindMax : LREAL;
	lrIntegAntiWindMin : LREAL;
	xEnIntegMax : BOOL;
	lrIntegMax  : LREAL;
	xEnIntegMin : BOOL;
	lrIntegMin  : LREAL;
	
	xEnKIAnti   : BOOL;
	lrKIAntiUp  : LREAL;   		// Integral ++ From Zero
	lrKIAntiMaxDown : LREAL; 	// Integral -- From Positive Value
	lrKIAntiDown  : LREAL; 		// Integral -- From Zero
	lrKIAntiMinUp : LREAL; 		// Integral ++ From Negative Value
	
	xEnIncompleteDiff : BOOL;
END_VAR
VAR_OUTPUT
	xBusy  : BOOL;
	xError : BOOL;
	
	lrSVPVe : LREAL;
	lrInteg : LREAL;
	lrDiff  : LREAL;	
	lrDiffRef  : LREAL;
	lrFD    : LREAL;
	lrFFD   : LREAL;
	
	lrBaseOP   : LREAL;
	lrOpV1 : LREAL;  // Calc P Value
	lrOpV2 : LREAL;  // Calc I Value
	lrOpV3 : LREAL;  // Calc D Value
	lrPVd  : LREAL;   // refPV - PV
	
	xIntegMaxAct : BOOL;
	xIntegMinAct : BOOL;
	xIntegAntiWindMaxAct : BOOL;
	xIntegAntiWindMinAct : BOOL;
	
END_VAR
VAR
	bFbState : E_FB_RUNST := E_FB_RUNST.unInit;
	
	lrPrePV   : LREAL;
	lrPrePVr  : LREAL;
	
	lrPreSVPVe : LREAL;
	lrPreInteg : LREAL;
	lrPreDiff  : LREAL;
	lrPreOP    : LREAL;
	
	lrError  : ARRAY[0..4] OF LREAL; // error, 0: cur
	
	lrTmpV1 : LREAL;
	lrTmpV2 : LREAL;
	lrTmpV3 : LREAL;
	
	lrCalcD  : LREAL;
	lrCalcFD : LREAL;
	lrCalcFFD : LREAL;
	lrCalcInteg : LREAL;
	lrCalcE    : LREAL;
	lrCalcAvgE : LREAL;
	lrCalcEr   : LREAL;
	
	xIntegMax : BOOL;
	xIntegMin : BOOL;
	xIntegDetach : BOOL;
	
	// PVr Calc
	lrCalcDiffRef : LREAL;
	lrCalcFDiffRef : LREAL;
	lrCalcFFDiffRef : LREAL;
	// PVr Calc
	
	lrKalmanK : LREAL;
	lrKalmanP : LREAL;
	FB_FilterKalman_0 : FB_FilterKalman;
	FB_FilterKalman_1 : FB_FilterKalman;
	
	xIntegClrTrig : BOOL := FALSE;
	xSvChangeTrig : BOOL := FALSE;
	R_TRIG_INTEG_CLR_0 : R_TRIG;
	R_TRIG_SV_CHANGE_0 : R_TRIG;
	
	i : INT;
	
END_VAR
//Init PID_Base FB
IF xEnable THEN
	IF E_FB_RUNST.unInit = bFbState THEN
		A_Init();
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

IF E_FB_RUNST.init = bFbState THEN
	lrPrePV  := lrPV;
	lrPrePVr := lrPVr;
	lrCalcD   := 0.0;
	lrCalcFD  := 0.0;
	lrCalcFFD := 0.0;
	lrCalcInteg := 0.0;
	lrCalcE := 0.0;
	lrCalcEr := 0.0;
END_IF


IF E_FB_RUNST.run <> bFbState THEN
	xBusy := TRUE;
ELSE
	xBusy := FALSE;
// Run PID_Base Calc
	lrError[0] := lrSV - lrPV;
	
// Current Error
	lrCalcE := lrError[0];
// Current Differential
	lrCalcD := lrPrePV - lrPV;

// OutTC Current Error
	// TODO:
// Current Differential
	lrCalcDiffRef := lrPrePVr - lrPVr;
	
	lrSVPVe := lrCalcE;
	lrPVd := lrPVr - lrPV;
	
// SV Change Reset Calc Value
	R_TRIG_SV_CHANGE_0(CLK:= xSvChanged, Q=> xSvChangeTrig);
	IF xSvChangeTrig THEN
//		xEnIntegClrSet := TRUE; 
		IF xSvChangeClrInteg THEN
			// Enable when PID first run or Process restart eg. BoatOut
			lrCalcInteg := 0.0;
		ELSE
			//
		END_IF
	END_IF
	
// Set Integ Calc Value
	R_TRIG_INTEG_CLR_0(CLK:= xEnIntegClrSet, Q=> xIntegClrTrig);
	IF xIntegClrTrig THEN
		lrCalcInteg := lrIntegClrSetV / (lrKI * lrCycleT);
	END_IF
	IF xEnIntegUpdate THEN
		lrCalcInteg := lrIntegUpdate / (lrKI * lrCycleT);
		//lrPreInteg := lrIntegUpdate;
	END_IF

// Calc Integral
	xIntegMaxAct := (lrTmpV2 > lrIntegMax) AND (lrCalcE > 0) AND (lrInteg > 0);
	xIntegAntiWindMaxAct := (lrBaseOP > lrIntegAntiWindMax) AND (lrCalcE > 0) AND (lrInteg > 0);
	xIntegMax := (xEnIntegMax AND xIntegMaxAct) OR (xEnIntegAntiWind AND xIntegAntiWindMaxAct);
	xIntegMinAct := (lrTmpV2 < lrIntegMin) AND (lrCalcE < 0) AND (lrInteg < 0);
	xIntegAntiWindMinAct := FALSE;
	xIntegMin := (xEnIntegMin AND xIntegMinAct) OR (xEnIntegAntiWind AND xIntegAntiWindMinAct);
	
	xIntegDetach := (lrCalcE > lrIntegDetachMax) OR (lrCalcE < lrIntegDetachMin) OR xForceIntegDetach;
	
	IF NOT (xIntegMax OR xIntegMin) THEN
		IF xIntegDetach AND xEnIntegDetach THEN
			IF ((lrCalcE < 0) AND (lrInteg > 1.0)) THEN
				lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiMaxDown;
				IF lrCalcInteg < 0 THEN
					lrCalcInteg := 0.0;
				END_IF
			END_IF
			IF ((lrCalcE > 0) AND (lrInteg < 0)) THEN
				lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiMinUp;
				IF lrCalcInteg > 0 THEN
					lrCalcInteg := 0.0;
				END_IF
			END_IF
	    ELSE
			IF lrCalcE < 0 THEN
				IF lrCalcInteg < 0 THEN
					lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiDown; //Integral --
				ELSE
					lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiMaxDown; //Integral --
				END_IF
				
			ELSE
				IF lrCalcInteg < 0 THEN
					lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiMinUp;   //Integral ++
				ELSE
					lrCalcInteg := lrCalcInteg + lrCalcE*lrKIAntiUp;   //Integral ++
				END_IF
				
			END_IF
		END_IF
	END_IF
	
// Push PV Error to Array and Calc Avg Value
	lrCalcAvgE := 0.0;
	FOR i := 1 TO 4 BY 1 DO
		lrError[i] := lrError[i-1];
		lrCalcAvgE := lrCalcAvgE + lrError[i];
	END_FOR
	lrCalcAvgE := lrCalcAvgE + lrError[0];
	lrCalcAvgE := lrCalcAvgE / 5.0;
	
// Filter For Diff
	FB_FilterKalman_0(
		xEnable:= (E_FB_RUNST.run = bFbState), 
		xReset:= FALSE, 
		lrInit_X:= lrCalcD, 
		lrInit_P:= 1.0, 
		lrInit_Q:= 0.015, 
		lrInit_R:= 0.5,   
		lrZ:= lrCalcD, 
		lrX=> lrCalcFD, 
		lrK=> lrKalmanK, 
		lrP=> lrKalmanP);
	
	//lrCalcFD := FC_FilterLowPass(k:= 10, rawV:= lrCalcD, filterV:= lrCalcFD);
	lrCalcFFD := FC_FilterLowPass(k:= 5, rawV:= lrCalcFD, filterV:= lrCalcFFD);
	
	lrFD := lrCalcFD;
	lrFFD := lrCalcFFD;
	
	// Calc PID Ctrl Op
	lrInteg := lrCalcInteg;
	IF xEnIncompleteDiff THEN
		lrDiff  := lrCalcFFD;  // Incomplete derivation
//		lrDiff  := lrCalcFD;  // Incomplete derivation
	ELSE
		lrDiff  := lrCalcD;
	END_IF
	
// Filter For DiffRef
	FB_FilterKalman_1(
		xEnable:= (E_FB_RUNST.run = bFbState), 
		xReset:= FALSE, 
		lrInit_X:= lrCalcDiffRef, 
		lrInit_P:= 1.0, 
		lrInit_Q:= 0.015, 
		lrInit_R:= 1.0,   
		lrZ:= lrCalcDiffRef, 
		lrX=> lrCalcFDiffRef, 
		lrK=> lrKalmanK, 
		lrP=> lrKalmanP);
	
	lrCalcFFDiffRef := FC_FilterLowPass(k:= 20, rawV:= lrCalcDiffRef, filterV:= lrCalcFFDiffRef);

	lrDiffRef := lrCalcFFDiffRef;

// *** PID Calc *** //
	
	lrTmpV1 := lrKP * lrCalcE;
	lrTmpV2 := lrKP * lrKI * lrCycleT * lrInteg;
	lrTmpV3 := lrKP * lrKD * lrDiff / lrCycleT;	
	
	lrBaseOP := lrTmpV1 + lrTmpV2 + lrTmpV3;
	
	lrOpV1 := lrTmpV1;
	lrOpV2 := lrTmpV2;
	lrOpV3 := lrTmpV3;
	
// Set Pre Value
	A_SetPreV();
END_IF


IF E_FB_RUNST.init = bFbState THEN
	bFbState := E_FB_RUNST.run;
END_IF