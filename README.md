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
| 第3档 | wc=1.5 wo=10 wr=3.5 nd=12 lrTauF=0.24 lrMVMax=60 | 到达2.8s / 1.2° / ±2°带2.6s | 待测（目标 3s/2°） |

## 五、已知边界（诚实条款）

- 孪生 v2 的 q 表为**稳态马力 q∞**，55~100% 段由 τe 模型外推（保持仅 0.4~0.7s）；
- 炉体热状态使同温度所需 MV 漂移 ±10%（40min 内实测），归 ESO 的 z2 实时吸收；
- 上/下行不对称（下行 θ≈0.2s，元件放热）未建模——下坡段孪生比实测掉得快；
- "到达 ≤3s"与"任何日子超调都 ≤2°"物理上不可兼得（滑行 = 进弯速度 × 滞后不确定性）；
- 高温静态锚点已实测到 480/500°C（7-16 标定场），线性散热模型到 500 大体成立；
- 设备温度上限 **500°C（瞬时可触、不可驻留；>480°C 段勿超 1 分钟）**，7-16 现场确认。
  控制/测试的安全预算据此制定：测试中止线按流程各自保守设定（450~470），超调后的瞬时峰值预算 ≤490。

数据说明：`user_feedback/` 与根目录的 `AIC9_DATA-*.csv` 为真机原始记录（10ms：time/PV1/PV2/SV/MV）；
7-2 的 xlsx 为另一硬件状态，仅存档。
