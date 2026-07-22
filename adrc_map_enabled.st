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
     [9] 可选 lrKd (默认 0=关闭, 2026-07-17): v 空间速度阻尼。lrWr 调大 (≥10, SV 平滑
         基本去掉) 后, 环路退化为欠阻尼二阶 (ζ=(1+kd)/(2√(wc·τ总)), τ总≈0.36s):
         wc=1.55 时首超 ~8°, wc 越大越糟 (wc=4 无阻尼可荡 40°)。lrKd 用 ESO 的速度
         估计 (lrB0·lrVDelayed + lrZ2, 即 ż1 的模型部分) 补回阻尼, 不对 PV 求导无噪声。
         这正是 PID 参考控制器里 D 项的角色 -- 但配合查表/z2/记账后可跨温区通用。
    [10] 可选 刹车预算律三件套 (默认全关, 2026-07-22; 目标 = 配方 kd 表退役):
         [10a] lrKb/lrBmin: 控制律加停车距离阻尼 −v̂·|v̂|/(2·max(−z2, lrBmin))。
               B̂=−z2 是"免费刹车"的在线读数, 随靶点/墙态/日漂自动变化。
               ⚠ 使能前置条件: 必须已换装"诚实表" aMapV (旧表中低段虚增 2.03×,
               7-22 断电初降速 17 点直测), 否则 −z2 单位虚大一倍 → 刹车距离
               低估 → 超调方向, 危险!
         [10b] lrBeta/lrTauEC/lrTauEH90: 元件温度代理 c_e 在线积分 (连续量) +
               冷元件功率如实记账。放电律已直测 (β=0.42 耦合形式, τ_c=103s);
               部分功率充电律待探针场验证, 之前 lrBeta 保持 0。
         包络认证 (analysis/cert10_envelope.py): 墙态/靶点自适应成立, 失败方向
         全部偏慢; 无在线信息时保守下限 4s 级, 接入 [10b]+机上拨盘后预期回到
         1.5~1.7s 级 (待 G5 上机验收)。
         v2 调节器 (7-22 换料) 配套: nDeadSteps 12→6 (θ_v2=0.06 脉冲场实测),
         aMapV 按 诚实×0.82 换装 (提案值待 4 点复核, 见 analysis/fit_element_0722.py)。
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

   第4档 (冲刺档, 2026-07-17, 目标 "1.5s级到400 / 首超≤3°"; 需 [9] lrKd, 对标 PID 参考
     控制器 7-17 晚实测 100→400 1.48s/3.0°):
     lrWc=4.5  lrWo=10.0  lrWr=30.0  nDeadSteps=12  lrTauF=0.24  lrKd=0.75  lrMVMax=90
     (lrKqHot=1.1 不动 -- 高 wo 下记账系数不敏感, 孪生验证 0.25/0.6/1.1 结果一致)
     孪生三炉况预期: 100→400 1.50~1.59s/≤2.8° | 200→400 1.30s/2.5° | 200→440 1.45s/3.2°
     整定拨盘: 首超>4° => lrKd +0.05; 不碰线变爬行 => lrKd -0.05; lrKd≥0.85 悬崖勿超
     诚实条款: 冷炉早晨首超仍 ~10° (机理同冷启动, 先热机); "任何日子都≤3°"依旧不可承诺
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
    lrKqHot    : LREAL := 1.1;    // [8] 记账温度修正斜率 (每100°C); 0=关闭; 投用 1.1
    lrKqTref   : LREAL := 315.0;  // [8] 缩水起始温度 [°C]
    lrKd       : LREAL := 0.0;    // [9] v 空间速度阻尼; 0=关闭(与旧版行为一致)。
                                  // lrWr 调大去掉 SV 平滑后环路失去阻尼 (欠阻尼二阶,
                                  // ζ=(1+kd)/(2√(wc·τ总)), τ总≈0.36s), 必须由本项补回。
                                  // 第4档投用 0.75; ≥0.85 过阻尼变爬行 (悬崖, 勿超)
    lrKb       : LREAL := 0.0;    // [10a] 刹车预算阻尼增益; 0=关闭。
                                  // ⚠ 前置条件: 必须已换装"诚实表"aMapV (见头部 [10] 说明),
                                  // 旧虚增表下 -lrZ2 读数偏大一倍 -> 刹车距离低估 -> 危险!
    lrBmin     : LREAL := 10.0;   // [10a] 刹车预算下限 [°C/s] (墙推/z2≥0 时的保护地板)
    lrBeta     : LREAL := 0.0;    // [10b] 元件冷启动冲幅度; 0=关闭; 7-22 实测 0.42 (耦合形式)
    lrTauEC    : LREAL := 103.0;  // [10b] 元件冷却时间常数 [s] (7-22 变间隔脉冲场实测)
    lrTauEH90  : LREAL := 0.35;   // [10b] 满幅充热时间常数 [s]; ⚠ 部分功率充电律
                                  // 尚未经专属激励验证 (待充电律探针场), 之前默认关闭
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
    lrVHat     : LREAL;                   // [9]/[10a]: ESO 的 PV 速度估计 = b0·vd + z2
    lrCe       : LREAL;                   // [10b]: 元件温度代理 ∈[0,1] (连续量, 非布尔)
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
        lrCe  := 0.0;     // [10b] 上电一律当元件全冷 (安全侧: 补偿偏多->偏慢不超调), τ_h 秒级自动收敛
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
    // [9] 速度阻尼: lrVHat = lrB0*lrVDelayed + lrZ2 = ESO 对 PV 速度的模型估计。
    //     不对 PV 求导, 无噪声放大。lrKd=0 时与旧版逐位一致。
    // [10a] 刹车预算阻尼 (默认 lrKb=0 关): 停车距离物理 v̂²/(2·B̂), B̂=max(−z2, Bmin)
    //     = 免费刹车的在线读数。诚实表前提下 z2 单位为真 °C/s (见头部警告)。
    lrU0   := lrWc * (lrRV1 - lrZ1);
    lrVHat := lrB0 * lrVDelayed + lrZ2;
    lrVCmd := lrU0 - lrZ2 - lrKd * lrVHat;
    IF lrKb > 0.0 THEN
        lrVCmd := lrVCmd - lrKb * lrVHat * ABS(lrVHat) / (2.0 * MAX(0.0 - lrZ2, lrBmin));
    END_IF
    lrVCmd := lrVCmd / lrB0;                            // heating-rate demand v [°C/s]

    IF xError THEN
        lrUCmd := lrVCmd;                               // 表非法: 直通 (等价现行无表版)
    ELSE
        lrVLo  := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMVMin);
        lrVHi  := FC_InterpLookup(aMapU, aMapV, nMapPoints, lrMVMax);
        lrVCmd := LIMIT(lrVLo, lrVCmd, lrVHi);          // 先按表值域夹紧, 再反查
        lrUCmd := FC_InterpInverse(aMapU, aMapV, nMapPoints, lrVCmd);   // -> real MV%
    END_IF
    lrMV := LIMIT(lrMVMin, lrUCmd, lrMVMax);

    // [10b] 元件温度代理更新 (连续量; 默认 lrBeta=0 关闭)
    IF lrBeta > 0.0 THEN
        IF lrMV > 2.0 THEN
            lrCe := lrCe + lrCycleT * (1.0 - lrCe) / (lrTauEH90 * 90.0 / MAX(lrMV, 5.0));
        ELSE
            lrCe := lrCe - lrCycleT * lrCe / lrTauEC;
        END_IF
        lrCe := LIMIT(0.0, lrCe, 1.0);
    END_IF

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
        IF lrBeta > 0.0 THEN
            // [10b] 冷元件多给的功率如实记账 -> z1/z2 冲刺期不再欠账
            lrVApplied := lrVApplied * (1.0 + lrBeta * (1.0 - lrCe));
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
    4) 第4档 (冲刺档) 上机: 须为含 [9] 的本版代码 (lrKd 生效); 下发第4档参数后
       100→400 / 200→400 / 200→440 各一次存 CSV; 按"整定拨盘"微调 lrKd (±0.05),
       回传数据做回放对账 (analysis/replay_identify.py)。
*)
END_FUNCTION_BLOCK
