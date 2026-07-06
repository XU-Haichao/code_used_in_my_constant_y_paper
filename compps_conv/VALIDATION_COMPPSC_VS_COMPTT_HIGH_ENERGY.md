# compPSc 与 compTT 高能谱验证

## 结论

采用 10 keV 所在的同一个能量 bin 分别归一化两个模型，并要求 10 keV 以上有效能段内的中位绝对相对误差和 P95 绝对相对误差均不超过 8%。本次扩展测试覆盖 90 组参数，其中 1 组通过该标准。

唯一通过的组合是：

- `kTe = 10 keV`
- `tau_compPS = 0.3`
- `tau_compTT = 0.15`，即 `0.5 * tau_compPS`
- compPSc 截断阶数：`max_scatter = 50`
- 中位绝对相对误差：1.24%
- P95 绝对相对误差：1.69%

整体趋势仍然是 `tau_compTT = 0.5 * tau_compPS` 最接近 compPSc。`tau_compTT = tau_compPS` 只在高温、高光深局部接近；`tau_compTT = 2 * tau_compPS` 整体明显更差。

## 测试设置

- 种子光子：`bbodyrad`，`kT = 0.005 keV`（5 eV）
- compPSc：本目录中的卷积模型，平板几何 `geom = 1`，`cosIncl = 0.5`
- compTT：XSPEC 内置模型，`approx = 1`，`T0 = 0.005 keV`
- 电子温度：`10, 20, 51.1, 100, 255.5 keV`
- compPSc 光深：`0.1, 0.3, 1, 2, 3, 5`
- compTT 光深映射：`tau_compTT = 0.5 * tau_compPS`、`tau_compTT = tau_compPS` 和 `tau_compTT = 2 * tau_compPS`
- compPSc 截断阶数：原生公式 `50 + int(4 tau^2)`；当 `tau >= 2` 时提高到 `max(原生值, 50 + int(30 tau^2))`
- 实际截断阶数：`tau = 0.1, 0.3, 1, 2, 3, 5` 分别对应 `50, 50, 54, 170, 320, 800`
- 能量网格：`0.001-1000 keV`，1200 个对数 bin
- 定标 bin：索引 800，`[10.0000, 10.1158) keV`，几何中心 `10.0577 keV`
- 比较范围：定标 bin 及以上；两个归一化谱均须为正且不低于各自高能峰值的 `1e-8`
- 残差定义：`compTT_normalized / compPSc_normalized - 1`

## 结果概览

下表给出每种温度和光深映射在六个 compPSc 光深上的误差范围。数值均为百分比。

| kTe (keV) | tau 映射 | 中位误差范围 | P95 误差范围 |
|---:|:---|---:|---:|
| 10 | `tau_TT = 0.5 tau_PS` | 0.88%-42.9% | 1.69%-152.4% |
| 10 | `tau_TT = 1 tau_PS` | 68.3%-655.7% | 166.3%-5421.4% |
| 10 | `tau_TT = 2 tau_PS` | 774.7%-3631.7% | 2015.2%-88462.3% |
| 20 | `tau_TT = 0.5 tau_PS` | 6.14%-47.3% | 14.6%-91.1% |
| 20 | `tau_TT = 1 tau_PS` | 156.1%-402.5% | 412.3%-3228.7% |
| 20 | `tau_TT = 2 tau_PS` | 813.6%-2197.8% | 2164.0%-44805.8% |
| 51.1 | `tau_TT = 0.5 tau_PS` | 13.5%-54.3% | 31.0%-62.6% |
| 51.1 | `tau_TT = 1 tau_PS` | 65.4%-184.4% | 275.3%-1452.3% |
| 51.1 | `tau_TT = 2 tau_PS` | 589.4%-833.7% | 1826.7%-14673.4% |
| 100 | `tau_TT = 0.5 tau_PS` | 16.5%-68.6% | 45.2%-75.8% |
| 100 | `tau_TT = 1 tau_PS` | 3.06%-94.3% | 105.3%-385.0% |
| 100 | `tau_TT = 2 tau_PS` | 157.0%-335.9% | 1016.0%-3190.7% |
| 255.5 | `tau_TT = 0.5 tau_PS` | 6.93%-80.3% | 55.5%-92.0% |
| 255.5 | `tau_TT = 1 tau_PS` | 7.57%-58.1% | 20.2%-74.1% |
| 255.5 | `tau_TT = 2 tau_PS` | 16.9%-152.0% | 201.8%-420.4% |

低温情况下的巨大 P95 误差来自高能指数尾部的谱形差异。当前统计会持续比较到任一归一化谱低于其高能峰值的 `1e-8`；因此结论严格对应这一预先规定的完整有效高能区间，而不是只比较 10 keV 附近的有限能带。

## 产物

- `validation_comppsc_vs_comptt_high_energy/summary.csv`：90 组汇总指标
- `validation_comppsc_vs_comptt_high_energy/spectra.csv`：逐能量 bin 的归一化谱和残差
- `validation_comppsc_vs_comptt_high_energy/metric_heatmaps.png`：误差热图
- `validation_comppsc_vs_comptt_high_energy/spectra_residuals_kTe_*_keV.png`：各温度的谱和残差图

运行脚本：

```bash
conda run -n heasoft_full python compps_conv/validate_comppsc_vs_comptt_high_energy.py
```

脚本仅在全部 90 组均通过时返回 0；本次科学验证只有 1 组通过，因此预期返回 1。
