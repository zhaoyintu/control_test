# AIC9 温控 ADRC：标定、数字孪生与开表控制

对 AIC9 加热炉（通道 1）的 ADRC（自抗扰）温度控制项目：用真机 bump 测试标定执行器特性
q(u)、纯滞后 θ 与元件热惯性 τe，构建 Python 数字孪生，在孪生上整定参数后再上真机验证。

- 交付控制代码：`adrc_map_enabled.st`（开查表版 FB_ADRC_Base，含插值函数与 lrTauF 升级）
- 完整操作流程文档：`analysis/theta_bump_test.html`（浏览器打开；标定怎么做、数据怎么用、参数怎么落地）

## 环境依赖

Python 3.10+，`numpy scipy pandas matplotlib`。所有命令在本目录下执行。

## 一、数字孪生怎么用

孪生 = `analysis/twin.py` + 标定参数 `analysis/twin_params.json`（当前为 v2，由
2026-07-15 bump session 整段拟合；v1 备份在 `twin_params_v1.json`）。

模型结构：`MV → 纯滞后θ → q(u)查表 → 一阶元件惯性τ2 → 净加热 − 散热h1·(T−c) → 积分 → PV`

### 1. 开环仿真（给 MV 序列，出温度轨迹）

```python
import numpy as np, sys; sys.path.insert(0, 'analysis')
from twin import Twin, DT

tw = Twin()                      # 读 analysis/twin_params.json
u  = np.r_[[14.0]*3000, [22.0]*200, [14.0]*2000]   # 10ms 一拍: 稳30s→+8%两秒→回
y  = tw.open_loop(u, y0=300.0)   # 返回同长度的 PV 轨迹
```

### 2. 闭环 ADRC 仿真（FB_ADRC_Base 逐行复刻 + 孪生对象）

```python
from twin import Twin, metrics
tw = Twin()
# 无表(u空间): b0=lrK; 开表(v空间): 见下
yy, uu = tw.closed_loop_adrc(b0=3.4, wc=2.0, wo=9.0, wr=3.0, nd=28,
                             y0=100.0, svt=400.0, T=20.0)
print(metrics(yy, 100.0, 400.0))   # (rise90[s], 超调[°C], settle±1°[s])
```

开表后的仿真在 v 空间做（控制器输出 = 加热速率指令，环路增益恒 1）：
把孪生的表替换成恒等映射、b0=1，见 `analysis/ladder_v2_scan.py` 里的 `vplant()` 写法；
带 lrTauF（ESO 元件惯性模型）的闭环参考 `ladder_v2_scan.py` 同款 `cl()` 实现。

### 3. 应力测试（体检合格 ≠ 只看标称）

整定新参数时，除标称孪生外必须过应力变体（增益 ×0.85/×1.15、θ+0.03、τ2×1.3），
写法见 `analysis/ladder_v2_scan.py`。经验：真机通常落在"标称 ~ 轻度应力"之间。

## 二、工具链（按工作流顺序）

| 脚本 | 用途 | 典型命令 |
|---|---|---|
| `analysis/theta_from_bump.py` | bump session 判读：θ、q 断点、稳态锚点，输出可粘贴 ST 变量表；`--write-twin` 粗校孪生 | `python3 analysis/theta_from_bump.py 数据.csv` |
| `analysis/fit_qinf_0715.py` | 整段拟合分离 q∞/τe/θ（精校，7-15 的参考实现） | 按文件内说明改路径后运行 |
| `analysis/twin_blindtest.py` | 孪生 vs 真机闭环盲测（v1/v2 对照） | `python3 analysis/twin_blindtest.py` |
| `analysis/ladder_v2_scan.py` | 开表后参数爬梯扫描（标称+应力） | `python3 analysis/ladder_v2_scan.py` |
| `analysis/replay_identify.py` | 回放辨识：确认真机某次阶跃实跑的参数 | `python3 analysis/replay_identify.py 数据.csv 180.3` |
| `analysis/test_drift_tolerance.py` | 判读脚本的漂移容忍自检（合成数据） | `python3 analysis/test_drift_tolerance.py` |
| `plots/make_plot_11_0715.py` 等 | 各分析图的生成脚本 | 运行后图落在 `plots/` |

## 三、标定 → 仿真 → 上机 的完整闭环

1. **标定**（详见 `analysis/theta_bump_test.html`）：闭环稳温 → 切手动只停 3~5s → 一步跳变
   → 回闭环；收尾 SV=450 稳 60s 收高温锚点。CSV 用 `theta_from_bump.py` 判读；
2. **孪生更新**：θ/τe/q∞ 用整段拟合复核后写入 `twin_params.json`（改前先备份）；
3. **仿真整定**：`ladder_v2_scan.py` 式扫描，标称合格 + 应力可接受才算候选；
4. **上机验证**：按档位爬梯（文件头与 HTML 落点③），每档一次 100→400 存 CSV；
5. **复核**：`replay_identify.py` 确认实跑参数 → 指标 vs 孪生对账，偏差 >20% 回到第 2 步。

## 四、当前参数档位（adrc_map_enabled.st 文件头同步维护）

| 档位 | 参数 | 预期 100→400 | 状态 |
|---|---|---|---|
| 第1档 | wc=1.0 wo=12 wr=3 nd=31 (lrTauF=0) | 2.7s / 0° / 5.4s | 7-15 真机验证通过 |
| 第2档 | wc=1.5 wo=8 wr=3 nd=24 (lrTauF=0) | 2.0s / 0.9° / 4.2s | 未测 |
| 第3档 | wc=1.5 wo=10 wr=3.5 nd=12 lrTauF=0.24 lrMVMax=60 | 到达2.8s / 1.2° / ±2°带2.6s | 7-15 晚实测 2.79s/2.5° |
| 第4档 | wc=4.5 wo=10 wr=30 nd=12 lrTauF=0.24 lrMVMax=90 + **lrKd 按配方下发** | 见下表 | 7-19 真机拨盘扫描锚定 |

**第4档生产 SOP（2026-07-19 真机数据锚定）**：基础卡固定，lrKd 随配方（SV）一起下发——
系统本就按配方下发参数，无需调度代码：

| 配方靶点 | lrKd | 实测锚定表现 |
|---|---|---|
| 400 类 | **0.85** | 1.71s / 1.0°（悬崖 0.95、超预算 0.75 各留一格距离） |
| 440/450 类 | **1.05**（求稳 1.1） | ~1.4s / ≈4°；1.1 为双模安全（1.6~3.3s / ≤3°） |
| 200 类 | **1.1** | 冷炉 0.99s / 2.6°；热炉体 4~6°（物理限制，墙温>靶温时无 ≤3° 解） |

现场唯一旋钮：首超 >4° → 该配方 lrKd +0.05；不碰线爬行 → −0.05。
配套规程：①阶跃前起点稳 ≥15s（根除台阶叠发）；②记录保温 MV 作免费炉态仪表。

## 五、已知边界（诚实条款）

- 孪生 v2 的 q 表为**稳态马力 q∞**，55~100% 段由 τe 模型外推（保持仅 0.4~0.7s）；
- 炉体热状态使同温度所需 MV 漂移 ±10%（40min 内实测），归 ESO 的 z2 实时吸收；
- 上/下行不对称（下行 θ≈0.2s，元件放热）未建模——下坡段孪生比实测掉得快；
- "到达 ≤3s"与"任何日子超调都 ≤2°"物理上不可兼得（滑行 = 进弯速度 × 滞后不确定性）；
- 7-17 到达极限判定【价格表部分已被当晚 [9] 修正，见下条】（`analysis/rung4_scan.py`、
  `plots/18_*.png`）：元件蓄热的滑行段刹不掉；旧律（低 wr + 无阻尼）下 MV 上限
  （真机冲刺峰值仅 53~58%，60→90 仿真无增益）与 lrK（v 空间恒 1，改了 = 让 ESO
  记假账）都不是速度旋钮；90% 满幅 380~425°C 实测 g≈0.80，孪生 v3 该段外推偏悲观。
- 7-17 晚修正（`analysis/pid_race_0717.py`、`plots/19_*.png`）：PID 对照场实测 100→400
  1.48s/3.0°（满幅 0.9s→断电@356→22% 接住），物理地板改判为**随炉况浮动 1.50~1.61s**
  （当晚保 400 需 22% MV，散热大 → 被动刹车强 → 地板低；PID 恰好踩着当晚地板）。
  wr 调大（≥10）去掉 SV 平滑后旧律失去阻尼（ζ=(1+kd)/(2√(wc·τ总))，光加 wc 越加越荡），
  须启用 [9] lrKd 速度阻尼（v̂=vd+z2，免微分）；第4档三炉况仿真 1.50~1.59s/首超≤2.8°。
  诚实条款照旧：冷炉早晨首超仍 ~10°（先热机），"任何日子都≤3°"不可承诺；
  τ2 失配是最大敏感项（+15% → 首超 ~12°，用 lrKd±0.05 拨盘吸收）；小阶跃（≤50°）
  为渐近碰线（1.7s 进 ±2° 带，严格触线偏慢），大阶跃指标不受影响。
- 高温静态锚点已实测到 480/500°C（7-16 标定场），线性散热模型到 500 大体成立；
- 设备温度上限 **500°C（瞬时可触、不可驻留；>480°C 段勿超 1 分钟）**，7-16 现场确认。
  控制/测试的安全预算据此制定：测试中止线按流程各自保守设定（450~470），超调后的瞬时峰值预算 ≤490。

数据说明：`user_feedback/` 与根目录的 `AIC9_DATA-*.csv` 为真机原始记录（10ms：time/PV1/PV2/SV/MV）；
7-2 的 xlsx 为另一硬件状态，仅存档。
