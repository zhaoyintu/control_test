import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# ---- validated categorical palette (light mode), fixed roles ----
C_SV  = '#4a3aa7'   # violet, dashed reference/target line
C_PV1 = '#2a78d6'   # blue
C_PV2 = '#1baf7a'   # aqua
C_MV  = '#eb6834'   # orange
GRID  = '#e1e0d9'
AXIS  = '#c3c2b7'
MUTED = '#898781'
INK   = '#0b0b0b'

def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.yaxis.label.set_color(INK)

def load(fname, sheet):
    df = pd.read_excel(fname, sheet_name=sheet, engine='openpyxl')
    df.columns = ['t', 'pv1', 'pv2', 'sv', 'mv']
    df['t'] = pd.to_datetime(df['t'])
    df['sec'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds()
    return df

def plot_overview(df, title, outpath, figsize=(11, 6)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [2.2, 1]})
    fig.patch.set_facecolor('#fcfcfb')
    ax1.set_facecolor('#fcfcfb')
    ax2.set_facecolor('#fcfcfb')

    ax1.step(df['sec'], df['sv'], where='post', color=C_SV, linestyle='--',
              linewidth=1.6, label='SV 设定值')
    ax1.plot(df['sec'], df['pv1'], color=C_PV1, linewidth=1.4, label='PV1 实测')
    ax1.plot(df['sec'], df['pv2'], color=C_PV2, linewidth=1.4, label='PV2 实测')
    style_axes(ax1, '温度 (℃)')
    ax1.legend(loc='best', frameon=False, fontsize=9.5, labelcolor=INK)
    ax1.set_title(title, color=INK, fontsize=13, loc='left', pad=10)

    ax2.plot(df['sec'], df['mv'], color=C_MV, linewidth=1.2, label='MV 输出功率')
    style_axes(ax2, 'MV 输出 (%)')
    ax2.set_xlabel('时间 (s，相对起点)', color=INK, fontsize=11)
    ax2.legend(loc='best', frameon=False, fontsize=9.5, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('saved', outpath)

def plot_zoom(df, t0, t1, title, outpath):
    seg = df[(df['sec'] >= t0) & (df['sec'] <= t1)]
    plot_overview(seg, title, outpath, figsize=(9, 5.4))

# ---------- 1) 数据分析.xlsx / data sheet: full ~96min run ----------
df1 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'data')
plot_overview(df1, '数据分析.xlsx / data — 开环特性测试（全程）',
              'plots/01_数据分析_data_全程.png', figsize=(13, 6))

# ---------- 2) 数据分析.xlsx / MV50% sheet: pulse test ----------
df2 = load('AIC9_DATA-20260702-185858_203502-数据分析.xlsx', 'MV50%')
plot_overview(df2, '数据分析.xlsx / MV50% — 50%功率脉冲开环测试',
              'plots/02_数据分析_MV50pct_脉冲测试.png', figsize=(11, 6))

# ---------- 3) PID闭环数据.xlsx / data: full closed-loop run ----------
df3 = load('AIC9_DATA-20260702-203502_204340-PID闭环数据.xlsx', 'data')
plot_overview(df3, 'PID闭环数据.xlsx — 闭环阶跃测试（全程，0→100→300→400→300→100→0℃）',
              'plots/03_PID闭环_全程.png', figsize=(13, 6))

# ---------- 4) zoom: 0->100C cold start (dead time + 47% overshoot) ----------
plot_zoom(df3, 0, 30, '冷启动阶跃 0→100℃ — 死区~13.8s + 超调47%（147.1℃）',
          'plots/04_PID闭环_zoom_冷启动0到100C.png')

# ---------- 5) zoom: 300->400C warm step (clean, <1% overshoot) ----------
plot_zoom(df3, 41, 54, '热态阶跃 300→400℃ — 死区<0.3s + 超调<1%（403.1℃）',
          'plots/05_PID闭环_zoom_热态300到400C.png')

print('all done')
