import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
file_path = 'TrainSet0507_NoDuplicates.xlsx'  # 필요시 경로 수정
df = pd.read_excel(file_path)

# 숫자 변환: 입력값
df['YS_num']  = pd.to_numeric(df['YS'],  errors='coerce')
df['UTS_num'] = pd.to_numeric(df['TS'],  errors='coerce')  # Excel의 TS → UTS
df['E_num']   = pd.to_numeric(df['E'],   errors='coerce') * 1000  # [GPa] → [MPa]
df['HB_num']  = pd.to_numeric(df['HB'],  errors='coerce')
# 숫자 변환: 실제 동적 파라미터
df['sf_num'] = pd.to_numeric(df['sf'], errors='coerce')
df['b_num']  = pd.to_numeric(df['b'],  errors='coerce')
df['ef_num'] = pd.to_numeric(df['ef'], errors='coerce')
df['c_num']  = pd.to_numeric(df['c'],  errors='coerce')

# 입력·출력 모두 유효한 행만 필터링
cols = ['YS_num','UTS_num','E_num','HB_num','sf_num','b_num','ef_num','c_num']
df_clean = df.dropna(subset=cols).reset_index(drop=True)

YS  = df_clean['YS_num'].values
UTS = df_clean['UTS_num'].values
E   = df_clean['E_num'].values
HB  = df_clean['HB_num'].values

# 2. predict_dynamic 함수 정의 (MATLAB 로직 이식)
def predict_dynamic(YSi, UTSi, Ei, HBi):
    UTSoYS = UTSi / YSi

    # Median Method
    sigma_M = 1.5 * UTSi
    b_M     = -0.090
    ef_M    = 0.450
    c_M     = -0.590

    # Hardness Method
    sigma_H = 4.25 * HBi + 225
    b_H     = -0.09
    ef_H    = (1/Ei) * (0.32*HBi**2 - 487*HBi + 191000)
    c_H     = -0.56

    # 그룹 분류 및 비율 계산
    if (182 <= HBi <= 352) and (457 <= YSi <= 1017) and (1.03 <= UTSoYS <= 1.7):
        A0, B0, b0, c0 = sigma_M/Ei, ef_M, b_M, c_M
        rA = -0.9113*UTSoYS**2 + 3.0596*UTSoYS - 2.4222
        rb = -0.8604*UTSoYS**2 + 3.3351*UTSoYS - 2.8642
        rB =  8.5755*UTSoYS**2 -26.5280*UTSoYS +20.5854
        rc =  0.8688*UTSoYS**2 - 2.6115*UTSoYS + 1.9371
        fr = (1.1, 0.3, 0.7, 1)

    elif (430 <= HBi <= 540) and (1488 <= YSi <= 1970) and (1.03 <= UTSoYS <= 1.13):
        A0, B0, b0, c0 = sigma_H/Ei, ef_H, b_H, c_H
        rA = -18.3335*UTSoYS**2 +37.7199*UTSoYS -19.3246
        rb =  68.4873*UTSoYS**2 -149.7633*UTSoYS +81.7085
        rB = 3758.8802*UTSoYS**2 -8311.3903*UTSoYS +4596.1684
        rc = 167.1381*UTSoYS**2 -362.1278*UTSoYS +196.3614
        fr = (1.0, 0.3, 1.5, 1)

    elif (312 <= HBi <= 442) and (989 <= YSi <= 1528) and (1.03 <= UTSoYS <= 1.7):
        A0, B0, b0, c0 = sigma_H/Ei, ef_H, b_H, c_H
        rA = -3.3616*UTSoYS**2 + 9.1707*UTSoYS - 5.9830
        rb = -1.5440*UTSoYS**2 + 4.8144*UTSoYS - 3.4938
        rB = 28.2302*UTSoYS**2 - 82.7563*UTSoYS +61.2820
        rc = -0.2491*UTSoYS**2 - 0.1318*UTSoYS + 0.7626
        fr = (1.5, 0.4, 0.75, 1)

    elif (HBi <= 258) and (YSi <= 560) and (UTSoYS <= 2):
        A0, B0, b0, c0 = sigma_M/Ei, ef_M, b_M, c_M
        rA = -0.5650*UTSoYS**2 + 1.7550*UTSoYS - 1.1975
        rb = -0.6509*UTSoYS**2 + 2.3801*UTSoYS - 1.9650
        rB = -0.4858*UTSoYS**2 + 1.3971*UTSoYS - 0.8997
        rc = -0.0414*UTSoYS**2 + 0.0897*UTSoYS - 0.1487
        fr = (1.2, 0.3, 0.85, 1)

    else:
        A0, B0, b0, c0 = sigma_M/Ei, ef_M, b_M, c_M
        rA = -0.5875*UTSoYS**2 + 3.6194*UTSoYS - 4.7258
        rb =  0.1457*UTSoYS**2 - 0.3200*UTSoYS + 0.4120
        rB = -4.2361*UTSoYS**2 +20.3259*UTSoYS -24.1341
        rc = -0.3917*UTSoYS**2 + 1.8995*UTSoYS - 2.4654
        fr = (1.2, 0.2, 1.5, 1)

    fA, fb, fB, fc = fr
    A = A0 * (1 + rA)
    b = b0 * (1 + rb)
    B = B0 * (1 + rB) if rB >= -1 else abs(rB)
    c = c0 * (1 + rc)

    sigma_f_pred = A * Ei
    b_pred       = b
    ef_pred      = B
    c_pred       = c

    return sigma_f_pred, b_pred, ef_pred, c_pred

# 3. 예측 수행
preds = np.array([predict_dynamic(YS[i], UTS[i], E[i], HB[i])
                  for i in range(len(df_clean))])
sigma_pred, b_pred, ef_pred, c_pred = preds.T

# 4. 실제값 추출
sigma_true = df_clean['sf_num'].values
b_true     = df_clean['b_num'].values
ef_true    = df_clean['ef_num'].values
c_true     = df_clean['c_num'].values

# 5. 시각화: 스타일 설정 및 그래프 생성
import matplotlib as mpl
import seaborn as sns

# Times New Roman 폰트 및 기타 스타일 설정
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 10,  # 기본 폰트 크기 조정
    'axes.labelsize': 12,  # 축 레이블 크기 조정
    'axes.titlesize': 12,  # 서브플롯 제목 크기 조정
    'legend.fontsize': 9,  # 범례 폰트 크기 조정
    'xtick.labelsize': 9,  # x축 눈금 레이블 크기 조정
    'ytick.labelsize': 9,  # y축 눈금 레이블 크기 조정
    'lines.linewidth': 1.5,  # 선 두께 조정
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,  # DPI 조정 (Mac Retina 디스플레이 고려)
    'axes.unicode_minus': False
})

# Color Universal Design safe 팔레트
palette = sns.color_palette("colorblind")

# 2×2 그래프 생성
fig, axs = plt.subplots(2, 2, figsize=(9, 8), facecolor='none') # figsize 조정
axs = axs.flatten()

params = [
    ("$\\sigma\'_f$", sigma_true, sigma_pred),
    ("$b$", b_true, b_pred),
    ("$\\varepsilon\'_f$", ef_true, ef_pred),
    ("$c$", c_true, c_pred),
]

for ax, (label, true, pred) in zip(axs, params):
    # b와 c 파라미터는 절대값 사용
    if label in ["$b$", "$c$"]:
        true = np.abs(true)
        pred = np.abs(pred)
    
    ratio = pred / true
    inside50 = np.mean((ratio >= 1/1.5) & (ratio <= 1.5)) * 100
    inside100 = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100

    mn = np.nanmin([true.min(), pred.min()]) * 0.9
    mx = np.nanmax([true.max(), pred.max()]) * 1.1
    mn = max(mn, 1e-6)  # 로그 스케일을 위한 최소값 설정
    xs = np.linspace(mn, mx, 100)

    # 산점도 및 기준선
    ax.scatter(true, pred, alpha=0.6, s=35, color=palette[0], edgecolor='k', linewidth=0.3) # 마커 크기 조정
    ax.plot(xs, xs, 'r-', linewidth=1.2, label='Ideal (y=x)') # 선 두께 조정
    ax.plot(xs, xs*1.5, 'g--', linewidth=1.0, label='±50%') # 선 두께 조정
    ax.plot(xs, xs/1.5, 'g--', linewidth=1.0) # 선 두께 조정
    ax.plot(xs, xs*2.0, 'b:', linewidth=1.0, label='±100%') # 선 두께 조정
    ax.plot(xs, xs*0.5, 'b:', linewidth=1.0) # 선 두께 조정

    # 로그 스케일 설정
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 레이블 및 제목 설정
    if label in ["$b$", "$c$"]:
        ax.set_xlabel(f'Actual |{label}|')
        ax.set_ylabel(f'Predicted |{label}|')
    else:
        ax.set_xlabel(f'Actual {label}')
        ax.set_ylabel(f'Predicted {label}')
    
    ax.set_title(f'{label}\nWithin ±50% band: {inside50:.1f}%\nWithin ±100% band: {inside100:.1f}%')
    
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor('none')

plt.tight_layout()
plt.show()