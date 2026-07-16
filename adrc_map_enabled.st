(* =====================================================================
   FB_ADRC_Base -- 开查表版完整代码 (2026-07-15)
   依据: AIC9_DATA-20260715-000131_004800 bump session 标定 (q∞ 表整段拟合)

   与真机在跑的版本 (user_feedback/adrc.st) 的差别 -- 共 6 处:
     [1] 查表四处恢复: FIFO 初始化 / 输出反查 / FIFO 记账 (原注释行), lrB0:=lrK 不变 (lrK 填 1.0)
     [2] VAR_INPUT 默认值 = 实测 q∞ 表 + 第1档参数 (变量表另行下发会覆盖, 默认值兜底)
     [3] v 指令先按表值域夹紧再反查 (防止表外外推), FIFO 记账用夹紧后的实际 v (抗积分饱和)
     [4] nDeadSteps 越界保护 (aVDelay 上限 [0..31], lrTheta 最大 0.31s)
     [5] 表单调性开机自检: 不单调 => xError 并退化为直通 (等价现行无表版, 不会失控)
     [6] 使能时 lrRV1 从 lrPV 起步 (原为 lrSV): 带着偏差投入时按 wr 平滑逼近, 无瞬间阶跃
     [7] 可选 lrTauF (默认 0=关闭, 行为与旧版逐拍一致): FIFO 出口加一阶滤波, 让 ESO 的
         对象模型含元件热惯性 τe -- 冲刺时 z2 不再把"元件还没烧起来"记成扰动假账, 刹车准。
         启用时 nDeadSteps 只补纯滞后 (12), 惯性由 lrTauF 补 (0.24)
     [8] 可选 lrKqHot (默认 0=关闭): FIFO 记账的温度修正。7-16 实测发现高段马力 (MV>30%)
         在高温缩水 (90%满幅实测比表低 15~20%), 不修正则 z2 冲刺记负账 -> 刹车吐账超调
         (440 实测 10°, 冷炉 400 也到 5~8°)。只修记账不修反查: 模型估错时方向必为
         "少给油早刹车" (孪生验证: 对象无缩水而修正常开 -> 仅偏慢 0.2° 超调)。
         投用值: lrKqHot=1.1, lrKqTref=315 (孪生 v3 拟合, 复现三场超调谱系)
   附带: FC_InterpLookup / FC_InterpInverse 两个函数 (工程里若已有同名实现, 删掉本文件中的即可)

   参数档位 (孪生 v2 仿真, 100→400):
     第1档(起步): lrWc=1.0  lrWo=12.0  lrWr=3.0  nDeadSteps=31   预期 2.7s/0°/5.4s, 全应力干净
                  [7-15 下午真机验证: 上升 2.1~2.9s / 超调 1~2.9° / settle 5.2~6.4s, 回放 RMSE 1.1% -- 通过]
     第2档(主推): lrWc=1.5  lrWo=8.0   lrWr=3.0  nDeadSteps=24   预期 2.0s/0.9°/4.2s
     第1档实测与预期吻合 (±20%) 再上第2档; 振铃或超调>2° => wc 退 1.2 或回第1档
     [警示: 7-15 下午实测 wc=2.0/wr=4 (跳过档位) => 超调 12~15°C -- 正是应力仿真预言的滞后角行为]

   第3档 (可选升级, 目标 "到达400≤3s 且超调≤2°"): 需启用 lrTauF (ESO 元件惯性模型, 见 [7])
     lrWc=1.5  lrWo=10.0  lrWr=3.5  nDeadSteps=12  lrTauF=0.24  lrMVMax=60
     [7-15 晚实测: 100→400 到达 2.79s/超调 2.5° -- 3s 目标达成; 但 200→440 超调 10°,
      7-16 冷炉复测 400 步超调 3.7~8.0° -- 根因为高段马力高温缩水, 见 [8]]

   第3档+ (2026-07-16, 在第3档上叠加记账温度修正):
     第3档全部参数不动, 另设 lrKqHot=1.1, lrKqTref=315
     孪生 v3 预期(暖炉): 100→400 2.6s/3.5° | 200→400 2.6s/2.4° | 200→440 2.7s/2.8° | 100→450 2.9s/3.4°
     (440/450 从 9~13° 压到 ~3°; 400 步剩余超调含冷炉壁效应, 冷启动首烧偏大 2~4° 属预期,
      可先用第1档热机 10 分钟再切换。设备上限已确认 500°C 瞬时/不驻留, 超调 ≤490 预算内全部安全)
   ===================================================================== *)

FUNCTION FC_InterpLookup : LREAL
// 单调折线表查值: y = g(x), 两端夹紧不外推
VAR_INPUT
    aX   : ARRAY[0..15] OF LREAL;
    aY   : ARRAY[0..15] OF LREAL;
    nPts : DINT;
    lrX  : LREAL;
END_VAR
VAR
    iLo, iHi, iMid, nP : DINT;
    lrDx : LREAL;
END_VAR

nP := LIMIT(2, nPts, 16);
IF lrX <= aX[0] THEN
    FC_InterpLookup := aY[0];
    RETURN;
END_IF
IF lrX >= aX[nP - 1] THEN
    FC_InterpLookup := aY[nP - 1];
    RETURN;
END_IF
iLo := 0;
iHi := nP - 1;
WHILE iHi - iLo > 1 DO
    iMid := (iLo + iHi) / 2;
    IF aX[iMid] <= lrX THEN
        iLo := iMid;
    ELSE
        iHi := iMid;
    END_IF
END_WHILE
lrDx := aX[iHi] - aX[iLo];
IF lrDx <= 0.0 THEN
    FC_InterpLookup := aY[iLo];      // 表退化保护
    RETURN;
END_IF
FC_InterpLookup := aY[iLo] + (aY[iHi] - aY[iLo]) * (lrX - aX[iLo]) / lrDx;
END_FUNCTION


FUNCTION FC_InterpInverse : LREAL
// 同一张表反查: x = g^-1(y)。要求 aY 严格递增 (FB 开机自检保证)
VAR_INPUT
    aX   : ARRAY[0..15] OF LREAL;
    aY   : ARRAY[0..15] OF LREAL;
    nPts : DINT;
    lrY  : LREAL;
END_VAR
VAR
    iLo, iHi, iMid, nP : DINT;
    lrDy : LREAL;
END_VAR

nP := LIMIT(2, nPts, 16);
IF lrY <= aY[0] THEN
    FC_InterpInverse := aX[0];
    RETURN;
END_IF
IF lrY >= aY[nP - 1] THEN
    FC_InterpInverse := aX[nP - 1];
    RETURN;
END_IF
iLo := 0;
iHi := nP - 1;
WHILE iHi - iLo > 1 DO
    iMid := (iLo + iHi) / 2;
    IF aY[iMid] <= lrY THEN
        iLo := iMid;
    ELSE
        iHi := iMid;
    END_IF
END_WHILE
lrDy := aY[iHi] - aY[iLo];
IF lrDy <= 0.0 THEN
    FC_InterpInverse := aX[iLo];     // 表退化保护
    RETURN;
END_IF
FC_InterpInverse := aX[iLo] + (aX[iHi] - aX[iLo]) * (lrY - aY[iLo]) / lrDy;
END_FUNCTION


FUNCTION_BLOCK FB_ADRC_Base
VAR_INPUT
    xEnable    : BOOL;
    xReset     : BOOL := FALSE;

    lrPV       : LREAL;    // measured PV, engineering units (degC)
    lrSV       : LREAL;    // setpoint, engineering units (degC)
    lrCycleT   : LREAL;    // execution period Ts (s) -- 10ms

    lrK        : LREAL := 1.0;    // 开查表后恒为 1.0 (v 空间增益已被表归一)
    lrTau      : LREAL := 13.0;   // [已废弃] 保留接口兼容, 内部不再使用
    lrWc       : LREAL := 1.0;    // controller bandwidth (rad/s)      -- 第1档 1.0 / 第2档 1.5
    lrWo       : LREAL := 12.0;   // observer bandwidth (rad/s)        -- 第1档 12  / 第2档 8
    lrWr       : LREAL := 3.0;    // reference-shaping bandwidth (rad/s)

    nDeadSteps : DINT := 31;      // 死区+元件惯性补偿拍数 = lrTheta/Ts -- 第1档 31 / 第2档 24
                                  // (上行纯滞后实测 0.12s, 元件热惯性 τe≈0.24s 一并补偿;
                                  //  启用 lrTauF 后只补纯滞后: 第3档 12)
    lrTauF     : LREAL := 0.0;    // ESO 元件惯性模型时间常数 [s]; 0=关闭(与旧版行为一致);
                                  // 第3档启用: 0.24 (实测 τe)
    lrKqHot    : LREAL := 0.0;    // [8] 记账温度修正斜率 (每100°C); 0=关闭; 投用 1.1
    lrKqTref   : LREAL := 315.0;  // [8] 缩水起始温度 [°C]
    nMapPoints : DINT := 9;       // breakpoints actually used in aMapU/aMapV
    // q∞ 实测表 (2026-07-15): MV% -> 稳态加热马力 [°C/s]。65% 以上平坦是实测特性。
    // 85% 点与 70% 实测同值(335), 平段会让反查病态, 已并入 70-100 直线段。
    aMapU      : ARRAY[0..15] OF LREAL := [0.0, 8.1, 12.3, 22.0, 30.0, 40.0, 55.0, 70.0, 100.0, 7(0.0)];
    aMapV      : ARRAY[0..15] OF LREAL := [0.0, 9.9, 23.4, 50.4, 85.2, 142.1, 248.4, 335.0, 345.7, 7(0.0)];

    xSvChanged : BOOL;            // [未使用] 保留接口兼容

    lrMVMin    : LREAL := 0.0;
    lrMVMax    : LREAL := 100.0;
END_VAR
VAR_OUTPUT
    xBusy  : BOOL;
    xError : BOOL;    // TRUE = 表不单调/点数非法, 已退化为直通 (lrUCmd := lrVCmd)

    lrMV   : LREAL;   // controller output, 0-100 %
    lrZ1   : LREAL;   // ESO state estimate (~PV)               -- diagnostic
    lrZ2   : LREAL;   // ESO total-disturbance estimate         -- diagnostic
    lrRV1  : LREAL;   // shaped reference trajectory            -- diagnostic
END_VAR
VAR
    bFbState : E_FB_RUNST := E_FB_RUNST.unInit;

    aVDelay    : ARRAY[0..31] OF LREAL;   // FIFO of recent v commands [°C/s]; [0] = newest
    lrB0       : LREAL;
    lrRV2      : LREAL;                   // shaped reference's own rate (internal to TD)
    lrRV1Old   : LREAL;
    lrErrO     : LREAL;
    lrU0       : LREAL;
    lrVCmd     : LREAL;
    lrVLo      : LREAL;                   // v 值域下限 = g(lrMVMin)
    lrVHi      : LREAL;                   // v 值域上限 = g(lrMVMax)
    lrUCmd     : LREAL;
    lrVApplied : LREAL;
    lrVDelayed : LREAL;
    lrVFilt    : LREAL;                   // lrTauF>0 时: FIFO 出口一阶滤波状态 (ESO 惯性模型)
    lrGHot     : LREAL;                   // lrKqHot>0 时: 记账温度修正系数
    nSteps     : DINT;                    // nDeadSteps 越界保护后的实际拍数

    i : DINT;
END_VAR

//Init ADRC FB
IF xEnable THEN
    IF E_FB_RUNST.unInit = bFbState THEN
        lrB0 := lrK;                      // 开表后 lrK=1.0 => v 空间增益恒 1

        // 表自检: 点数 2..16 且 U/V 均严格递增, 否则退化为直通
        xError := (nMapPoints < 2) OR (nMapPoints > 16);
        IF NOT xError THEN
            FOR i := 1 TO nMapPoints - 1 DO
                IF (aMapU[i] <= aMapU[i-1]) OR (aMapV[i] <= aMapV[i-1]) THEN
                    xError := TRUE;
                END_IF
            END_FOR
        END_IF

        FOR i := 0 TO 31 DO
            aVDelay[i] := FC_InterpLookup(aMapU, aMapV, nMapPoints, 0.0);
        END_FOR
        lrVFilt := aVDelay[0];
        lrZ1  := lrPV;
        lrZ2  := 0.0;
        lrRV1 := lrPV;    // 从当前温度起步: 带偏差投入时按 lrWr 平滑逼近 SV, 无瞬间阶跃
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

    lrB0   := lrK;
    nSteps := LIMIT(0, nDeadSteps, 31);   // FIFO 上限 [0..31] => lrTheta 最大 0.31s

    // === 1) reference shaping (critically damped 2nd order tracking filter) ===
    // Decouples "how fast we approach a new SV" from "how fast the loop rejects
    // disturbance". Tune lrWr against an explicit overshoot budget.
    lrRV1Old := lrRV1;
    lrRV1 := lrRV1 + lrCycleT * lrRV2;
    lrRV2 := lrRV2 + lrCycleT * (-2.0*lrWr*lrRV2 - lrWr*lrWr*(lrRV1Old - lrSV));

    // === 2) ESO: reconcile estimate with this scan's measurement ===
    // Uses the delayed v (nSteps old), not the just-committed one -- this is what
    // "vanilla ADRC applied to a plant with known dead time" should actually do.
    lrVDelayed := aVDelay[nSteps];
    IF lrTauF > 0.0 THEN
        // [7] ESO 对象模型升级: 纯滞后之外再含一阶元件惯性 τe,
        //     冲刺/刹车时 z2 不再把元件的爬升/放热记成扰动假账
        lrVFilt := lrVFilt + lrCycleT * (lrVDelayed - lrVFilt) / lrTauF;
        lrVDelayed := lrVFilt;
    END_IF
    lrErrO := lrPV - lrZ1;
    lrZ1 := lrZ1 + lrCycleT * (lrZ2 + lrB0*lrVDelayed + 2.0*lrWo*lrErrO);
    lrZ2 := lrZ2 + lrCycleT * (lrWo*lrWo*lrErrO);

    // === 3) control law: proportional feedback on the ESO's clean estimate,
    //        cancel the estimated total disturbance, invert the measured q(u) map ===
    lrU0   := lrWc * (lrRV1 - lrZ1);
    lrVCmd := (lrU0 - lrZ2) / lrB0;                     // heating-rate demand v [°C/s]

    IF xError THEN
        lrUCmd := lrVCmd;                               // 表非法: 直通 (等价现行无表版)
    ELSE
        lrVLo  := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMVMin);
        lrVHi  := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMVMax);
        lrVCmd := LIMIT(lrVLo, lrVCmd, lrVHi);          // 先按表值域夹紧, 再反查
        lrUCmd := FC_InterpInverse(aMapU, aMapV, nMapPoints, lrVCmd);   // -> real MV%
    END_IF
    lrMV := LIMIT(lrMVMin, lrUCmd, lrMVMax);

    // === 4) advance the delay FIFO with what v ACTUALLY happened (post-clamp!) ===
    IF xError THEN
        lrVApplied := lrMV;
    ELSE
        lrVApplied := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMV);
        IF (lrKqHot > 0.0) AND (lrPV > lrKqTref) AND (lrMV > 30.0) THEN
            // [8] 记账温度修正: 高温高段实际马力缩水, 账本按实际记, z2 不再有冲刺负账。
            //     只修记账不修反查 -- 模型估错时方向必为"少给油早刹车"(安全侧)
            lrGHot := 1.0 - lrKqHot * LIMIT(0.0, (lrMV - 30.0) / 30.0, 1.0)
                            * (lrPV - lrKqTref) / 100.0;
            IF lrGHot < 0.4 THEN
                lrGHot := 0.4;
            END_IF
            lrVApplied := lrVApplied * lrGHot;
        END_IF
    END_IF
    FOR i := 31 TO 1 BY -1 DO
        aVDelay[i] := aVDelay[i-1];
    END_FOR
    aVDelay[0] := lrVApplied;

END_IF

IF E_FB_RUNST.init = bFbState THEN
    bFbState := E_FB_RUNST.run;
END_IF

(*
  单位一致性备忘:
    aMapV / lrVCmd / aVDelay / lrZ2 的量纲都是加热速率 [°C/s];
    lrZ2 是 ESO 估计的"总扰动"(散热 + 表误差 + 慢漂), 稳态时 lrVCmd = -lrZ2 恰好托住散热。
  投运检查顺序 (详见 analysis/theta_bump_test.html 落点①~③):
    1) 表和参数下发后先看 xError = FALSE;
    2) 第1档跑 100→400 存 CSV: 预期 2.7s 升 / 0° 超调 / 5.4s 收敛;
    3) 吻合 ±20% 再上第2档 (wc=1.5, wo=8, nDeadSteps=24)。
*)
END_FUNCTION_BLOCK
