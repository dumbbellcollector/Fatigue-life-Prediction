import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
from sklearn.metrics import r2_score
import matplotlib as mpl
import seaborn as sns

# 1. 데이터 로드 및 전처리
file_path = 'TrainSet0507_NoDuplicatesHV.xlsx'  # 필요시 경로 수정
df = pd.read_excel(file_path)

# 숫자 변환: 입력값
df['E_num']  = pd.to_numeric(df['E'],  errors='coerce')
df['YS_num']  = pd.to_numeric(df['YS'],  errors='coerce')
df['UTS_num'] = pd.to_numeric(df['TS'],  errors='coerce')  # Excel의 TS → UTS
df['HB_num']  = pd.to_numeric(df['HB'],  errors='coerce')
# 숫자 변환: 실제 동적 파라미터
df['sf_num'] = pd.to_numeric(df['sf'], errors='coerce')
df['b_num']  = pd.to_numeric(df['b'],  errors='coerce')
df['ef_num'] = pd.to_numeric(df['ef'], errors='coerce')
df['c_num']  = pd.to_numeric(df['c'],  errors='coerce')

# 입력·출력 모두 유효한 행만 필터링
cols = ['YS_num','UTS_num','HB_num','sf_num','b_num','ef_num','c_num']
df_clean = df.dropna(subset=cols).reset_index(drop=True)

E  = df_clean['E_num'].values
YS  = df_clean['YS_num'].values
UTS = df_clean['UTS_num'].values
HB  = df_clean['HB_num'].values

# 2. predict_dynamic 함수 정의 (MATLAB 로직 이식)
def predict_dynamic(Ei, YSi, UTSi, HBi):
    """
    새로운 경도 기반 변환식
    ef = (0.32(HB)^2 - 487(HB) + 191000) / E
    sf = (4.25HB + 225) / E
    b = -0.09
    c = -0.56
    """
    # 탄성계수 (MPa 단위)
    E = Ei
    
    # 새로운 변환식 적용
    ef_Dev = (0.32 * HBi**2 - 487 * HBi + 191000) / (1000*E)
    sf_Dev = (4.25 * HBi + 225)
    b_Dev = -0.09
    c_Dev = -0.56
    
    return sf_Dev, b_Dev, ef_Dev, c_Dev

# 3. 예측 수행
preds = np.array([predict_dynamic(E[i], YS[i], UTS[i], HB[i])
                  for i in range(len(df_clean))])
sigma_pred, b_pred, ef_pred, c_pred = preds.T

# 4. 실제값 추출
sigma_true = df_clean['sf_num'].values
b_true     = df_clean['b_num'].values
ef_true    = df_clean['ef_num'].values
c_true     = df_clean['c_num'].values

# 5. 피로 수명(2Nf) 비교 (Basan 논문 방식)
print("\n=== 피로 수명(2Nf) 비교 분석 ===")

# 평가를 위한 변형률 진폭(Δε/2) 값 정의
defined_total_strain_amplitudes = np.array([
    0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.009, 0.015
])
print(f"평가를 위해 사용할 고정된 총 변형률 진폭 (Δε/2 또는 ε_a) 값들: {defined_total_strain_amplitudes}")

def solve_2Nf_from_strain(params, E_val, epsilon_a_target):
    spf, b_exp, epf, c_exp = params
    E_val_safe = max(float(E_val), 1e-9)

    def equation(two_Nf_val_log10):
        two_Nf_val = 10**two_Nf_val_log10
        if two_Nf_val <= 0: return float('inf')
        try:
            term1 = (spf / E_val_safe) * np.power(max(two_Nf_val, 1e-9), b_exp)
            term2 = epf * np.power(max(two_Nf_val, 1e-9), c_exp)
            return term1 + term2 - epsilon_a_target
        except OverflowError: return float('inf')
        except ValueError: return float('inf')

    initial_guess_log10 = 4.0
    if epsilon_a_target > 0.01: initial_guess_log10 = 2.0
    elif epsilon_a_target < 0.002: initial_guess_log10 = 5.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solution_log10, infodict, ier, mesg = fsolve(equation, initial_guess_log10, full_output=True, xtol=1e-7, maxfev=500)
    
    if ier == 1 and isinstance(solution_log10, (np.ndarray, list)) and len(solution_log10) > 0:
        return 10**solution_log10[0]
    elif ier == 1 and isinstance(solution_log10, (int, float)):
         return 10**solution_log10
    else:
        return np.nan

# 필요한 데이터 준비
# 예측된 파라미터와 실제 파라미터를 배열로 변환
predicted_params_all_samples_np = np.column_stack([sigma_pred, b_pred, ef_pred, c_pred])
true_params_all_samples_np = np.column_stack([sigma_true, b_true, ef_true, c_true])

# 탄성계수 추정 (간단한 추정식 사용)
# 일반적으로 강재의 탄성계수는 약 200 GPa
E_values_all_samples_np = np.full(len(df_clean), 200000)  # MPa 단위

# 각 샘플 및 각 변형률 진폭 레벨에 대해 2Nf_exp 및 2Nf_est 계산
twoNf_exp_collected = []
twoNf_est_collected = []

num_test_samples = len(true_params_all_samples_np)
print(f"\nCalculating 2Nf_exp and 2Nf_est for {num_test_samples} test samples using {len(defined_total_strain_amplitudes)} strain levels...")

for i in range(num_test_samples):
    params_true_sample = true_params_all_samples_np[i, :]
    params_pred_sample = predicted_params_all_samples_np[i, :]
    E_val_sample_i = E_values_all_samples_np[i]

    if any(np.isnan(params_true_sample)) or any(np.isnan(params_pred_sample)):
        continue
        
    for strain_amplitude_level in defined_total_strain_amplitudes:
        twoNf_exp = solve_2Nf_from_strain(params_true_sample, E_val_sample_i, strain_amplitude_level)
        twoNf_est = solve_2Nf_from_strain(params_pred_sample, E_val_sample_i, strain_amplitude_level)

        if not np.isnan(twoNf_exp) and not np.isnan(twoNf_est) and \
           twoNf_exp > 0 and twoNf_est > 0:
            if 10**1 <= twoNf_exp <= 10**7 :
                twoNf_exp_collected.append(twoNf_exp)
                twoNf_est_collected.append(twoNf_est)

log_2Nf_exp_final = np.log10(np.array(twoNf_exp_collected))
log_2Nf_est_final = np.log10(np.array(twoNf_est_collected))

print(f"Successfully calculated and filtered {len(log_2Nf_exp_final)} (2Nf_exp, 2Nf_est) pairs for plotting.")

# 산점도 (Scatter Plot) 작성
if len(log_2Nf_exp_final) > 1:
    # 피로 수명 비교를 위한 스타일 설정 (글자 크기 조정)
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': 'Times New Roman', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 11, 'legend.fontsize': 9,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1.5,
        'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 100,
        'axes.unicode_minus': False
    })
    
    plt.figure(figsize=(8, 7), facecolor='none')
    
    r2_2Nf_final = r2_score(log_2Nf_exp_final, log_2Nf_est_final)
    
    # Scatter band inclusion rate 계산 (1.3x, 2x, 3x, 5x, 10x)
    ratio_2Nf_final = np.array(twoNf_est_collected) / np.array(twoNf_exp_collected)
    inside_1_3x_2Nf_final = np.logical_and(ratio_2Nf_final >= 1/1.3, ratio_2Nf_final <= 1.3).mean() * 100
    inside_2x_2Nf_final = np.logical_and(ratio_2Nf_final >= 1/2, ratio_2Nf_final <= 2).mean() * 100
    inside_3x_2Nf_final = np.logical_and(ratio_2Nf_final >= 1/3, ratio_2Nf_final <= 3).mean() * 100
    inside_5x_2Nf_final = np.logical_and(ratio_2Nf_final >= 1/5, ratio_2Nf_final <= 5).mean() * 100
    inside_10x_2Nf_final = np.logical_and(ratio_2Nf_final >= 1/10, ratio_2Nf_final <= 10).mean() * 100
    
    plot_axis_min_log = 1.0
    plot_axis_max_log = 7.0
    line_vals_log = np.linspace(plot_axis_min_log, plot_axis_max_log, 100)

    scatter_color = 'blue'

    plt.scatter(log_2Nf_exp_final, log_2Nf_est_final, alpha=0.4, s=30, 
                color=scatter_color, edgecolor='k', linewidth=0.2)
    
    plt.plot(line_vals_log, line_vals_log, 'r-', linewidth=1.5, label='Ideal (y=x)')
    
    # Scatter band 플롯 (2x, 3x, 5x, 10x)
    log_factor_2 = np.log10(2.0)
    log_factor_3 = np.log10(3.0)
    log_factor_5 = np.log10(5.0)
    log_factor_10 = np.log10(10.0)
    
    # 2x Band
    plt.plot(line_vals_log, line_vals_log + log_factor_2, color='darkorange', linestyle='--', linewidth=1.2, label='±2x Band')
    plt.plot(line_vals_log, line_vals_log - log_factor_2, color='darkorange', linestyle='--', linewidth=1.2)
    
    # 3x Band
    plt.plot(line_vals_log, line_vals_log + log_factor_3, color='green', linestyle=':', linewidth=1.2, label='±3x Band')
    plt.plot(line_vals_log, line_vals_log - log_factor_3, color='green', linestyle=':', linewidth=1.2)
    
    # 5x Band
    plt.plot(line_vals_log, line_vals_log + log_factor_5, color='purple', linestyle='-.', linewidth=1.2, label='±5x Band')
    plt.plot(line_vals_log, line_vals_log - log_factor_5, color='purple', linestyle='-.', linewidth=1.2)
    
    # 10x Band
    plt.plot(line_vals_log, line_vals_log + log_factor_10, color='brown', linestyle='-', linewidth=1.0, label='±10x Band')
    plt.plot(line_vals_log, line_vals_log - log_factor_10, color='brown', linestyle='-', linewidth=1.0)

    plt.xlabel('Load Reversals(experimental), Log(2N$_{f,exp}$)')
    plt.ylabel('Load Reversals(estimated), Log(2N$_{f,est}$)')
    
    title_text = (f'Fatigue Life (2Nf) Prediction Comparison\n'
                  f'(Calculated at predefined $\\Delta\\epsilon/2$ levels)\n'
                  f'Within ±2x band: {inside_2x_2Nf_final:.1f}%, '
                  f'Within ±3x band: {inside_3x_2Nf_final:.1f}%, '
                  f'Within ±5x band: {inside_5x_2Nf_final:.1f}%, '
                  f'Within ±10x band: {inside_10x_2Nf_final:.1f}%')
    plt.title(title_text)
    
    plt.xlim(plot_axis_min_log - 0.2, plot_axis_max_log + 0.2)
    plt.ylim(plot_axis_min_log - 0.2, plot_axis_max_log + 0.2)
    
    tick_values = np.arange(int(np.floor(plot_axis_min_log)), int(np.ceil(plot_axis_max_log)) + 1, 1.0)
    plt.xticks(tick_values)
    plt.yticks(tick_values)
    
    plt.legend(frameon=True, loc='upper left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('none')

    plt.tight_layout()
    plt.show()
    
    print(f"\nFatigue Life (log10(2Nf)) R2 Score: {r2_2Nf_final:.4f}")
    print(f"Percentage of predictions within ±1.3x scatter band: {inside_1_3x_2Nf_final:.2f}%")
    print(f"Percentage of predictions within ±2x scatter band: {inside_2x_2Nf_final:.2f}%")
    print(f"Percentage of predictions within ±3x scatter band: {inside_3x_2Nf_final:.2f}%")
    print(f"Percentage of predictions within ±5x scatter band: {inside_5x_2Nf_final:.2f}%")
    print(f"Percentage of predictions within ±10x scatter band: {inside_10x_2Nf_final:.2f}%")

else:
    print("No valid 2Nf data points to plot after all calculations and filtering.")

print("\n=== 분석 완료 ===")