# Cell 12: GUI 고려사항 (Streamlit - Modified for Hybrid Output)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd

# --- 모델 정의 복제 (FatiguePINN) ---
class FatiguePINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128], dropout_p=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p # 드롭아웃 비율 저장

        layers = []
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            # 드롭아웃 레이어 추가
            if self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
            last_dim = hidden_dim
        
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Empirical Method Functions ---
HB_CLASS1_MAX = 150; HB_CLASS2_MAX = 500; HB_CLASS3_MAX = 700

def modified_universal_slopes(ts_mpa, e_mpa):
    """Modified Universal Slopes Method"""
    spf = 1.9 * ts_mpa
    epf = 0.623 * (ts_mpa / e_mpa) ** 0.832
    return spf, epf

def uniform_material_law(ts_mpa, e_mpa):
    """Uniform Material Law"""
    if ts_mpa < 750:  # Low strength steel
        spf = 1.5 * ts_mpa
        epf = 0.59 * (ts_mpa / e_mpa) ** 0.58
    else:  # High strength steel
        spf = 1.67 * ts_mpa
        epf = 0.35
    return spf, epf

def hardness_method(hb, ts_mpa, e_mpa):
    """Hardness Method"""
    spf = 4.25 * hb + 225
    epf = 0.32 * (hb / 1000) ** (-1.24)
    return spf, epf

method_map = {1: hardness_method, 2: hardness_method, 3: hardness_method, 4: hardness_method}
method_names = {1: "Hardness Method", 2: "Hardness Method", 3: "Hardness Method", 4: "Hardness Method"}

def get_physics_params(hb_val, ts_mpa_val, e_mpa_val):
    """Get physics-based parameters based on material class"""
    method_name = "Unknown"
    spf_physics = epf_physics = None
    
    if hb_val <= 0 or np.isnan(hb_val):
        return np.nan, np.nan, "Invalid HB value"
    
    # Determine material class based on HB
    if hb_val <= HB_CLASS1_MAX:
        material_class = 1
    elif hb_val <= HB_CLASS2_MAX:
        material_class = 2
    elif hb_val <= HB_CLASS3_MAX:
        material_class = 3
    else:
        material_class = 4
    
    # Get method function and name
    method_func = method_map.get(material_class)
    method_name = method_names.get(material_class, "Unknown")
    
    if method_func:
        try:
            spf_physics, epf_physics = method_func(hb_val, ts_mpa_val, e_mpa_val)
        except Exception as e:
            return np.nan, np.nan, f"Error in {method_name}: {str(e)}"
    
    return spf_physics, epf_physics, method_name

# --- 예측 함수 복제 (predict_fatigue_curves_hybrid) ---
def predict_fatigue_curves_hybrid(E_val, YS_val, TS_val, HB_val, model, scaler_X, scaler_y, device, nu=0.3):
    model.eval()
    input_features = np.array([[E_val, YS_val, TS_val, HB_val]])
    input_scaled = scaler_X.transform(input_features)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predicted_tensile_params_scaled = model(input_tensor)
    
    predicted_tensile_params_orig = scaler_y.inverse_transform(predicted_tensile_params_scaled.cpu().numpy())[0]
    tensile_target_cols_local = ['spf_MPa', 'b', 'epf', 'c']
    
    # 물리 기반 spf, epf 계산
    spf_physics, epf_physics, method_name = get_physics_params(HB_val, TS_val, E_val)
    
    # 신경망 예측값 가져오기
    nn_params = {name: val for name, val in zip(tensile_target_cols_local, predicted_tensile_params_orig)}
    
    # 하이브리드 파라미터 설정 (spf, epf는 물리 기반, b, c는 신경망)
    tensile_params = {
        'spf_MPa': spf_physics if not np.isnan(spf_physics) else nn_params['spf_MPa'],
        'epf': epf_physics if not np.isnan(epf_physics) else nn_params['epf'],
        'b': nn_params['b'],
        'c': nn_params['c'],
        'estimation_method_spf_epf': method_name
    }
    
    # 인장 곡선 계산
    spf_prime = tensile_params['spf_MPa']
    b = tensile_params['b']
    epf_prime = tensile_params['epf']
    c = tensile_params['c']
    
    reversals = np.logspace(1, 7, num=100)
    E_val_safe = max(E_val, 1e-6)
    
    elastic_strain_en = (spf_prime / E_val_safe) * (reversals ** b)
    plastic_strain_en = epf_prime * (reversals ** c)
    strain_amplitude_en = elastic_strain_en + plastic_strain_en
    
    # 전단 파라미터 변환
    shear_params = {}
    conversion_method = "Unknown"
    
    # von Mises 변환
    tau_vm = spf_prime / np.sqrt(3)
    gamma_vm = np.sqrt(3) * epf_prime
    
    # Max Principal 변환
    tau_mp = spf_prime / (1 + nu)
    gamma_mp = 2 * epf_prime
    
    if TS_val <= 1100:
        tauf_prime, gammaf_prime = tau_vm, gamma_vm
        conversion_method = "von Mises Criteria"
        b0 = b
        c0 = c
    elif TS_val >= 1696:
        tauf_prime, gammaf_prime = tau_mp, gamma_mp
        conversion_method = "Maximum Principal Criteria"
        b0 = b
        c0 = c
    else:
        α = (TS_val - 1100) / (1696 - 1100)
        tauf_prime = (1-α)*tau_vm + α*tau_mp
        gammaf_prime = (1-α)*gamma_vm + α*gamma_mp
        conversion_method = f"Interpolated (α={α:.2f})"
        b0 = b
        c0 = c
    
    shear_params['tauf_MPa'] = tauf_prime if tauf_prime is not None else np.nan
    shear_params['gammaf'] = gammaf_prime if gammaf_prime is not None else np.nan
    shear_params['b0'] = b0 if b0 is not None else np.nan
    shear_params['c0'] = c0 if c0 is not None else np.nan
    shear_params['conversion_method'] = conversion_method
    
    # 전단 곡선 계산
    G_val = E_val_safe / (2 * (1 + nu))
    elastic_shear_strain_gn = np.full_like(reversals, np.nan)
    plastic_shear_strain_gn = np.full_like(reversals, np.nan)
    strain_amplitude_gn = np.full_like(reversals, np.nan)
    
    if not any(np.isnan(val) for key, val in shear_params.items() if key != 'conversion_method'):
        elastic_shear_strain_gn = (tauf_prime / G_val) * (reversals ** b0)
        plastic_shear_strain_gn = gammaf_prime * (reversals ** c0)
        strain_amplitude_gn = elastic_shear_strain_gn + plastic_shear_strain_gn
    
    return (tensile_params, reversals, strain_amplitude_en, elastic_strain_en, plastic_strain_en,
            shear_params, reversals, strain_amplitude_gn, elastic_shear_strain_gn, plastic_shear_strain_gn)

# --- Load Model and Scalers ---
@st.cache_resource
def load_resources(model_path='best_fatigue_pinn_model.pth',
                   scaler_x_path='scaler_X.pkl',
                   scaler_y_path='scaler_y.pkl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    paths_exist = True
    required_files = [model_path, scaler_x_path, scaler_y_path]
    for f_path in required_files:
        if not os.path.exists(f_path):
            st.error(f"Error: Required file not found: {f_path}")
            paths_exist = False
    if not paths_exist: return None, None, None, None

    try:
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        input_dim = scaler_X.n_features_in_
        output_dim_tensile = scaler_y.n_features_in_

        # --- main.ipynb와 동일한 hidden_dims 및 dropout_p 사용 ---
        model_hidden_dims = [128, 256, 128] # main.ipynb에서 사용한 층 구조
        model_dropout_p = 0.2               # main.ipynb에서 사용한 드롭아웃 비율

        # dropout_p 인자를 포함하여 모델 인스턴스화
        model = FatiguePINN(input_dim, output_dim_tensile, 
                            hidden_dims=model_hidden_dims, 
                            dropout_p=model_dropout_p).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # 드롭아웃은 추론 시에는 비활성화됨 (eval() 모드에서 자동 처리)
        return model, scaler_X, scaler_y, device
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('Fatigue Life Predictor (ε-N / γ-N)')
st.write("Enter Monotonic Properties and Select mode, Get Prediction.")

# 리소스 로드
model, scaler_X, scaler_y, device = load_resources()

if model is None: st.stop()

# 사이드바에 입력 섹션 배치
with st.sidebar:
    st.header("입력 파라미터")
    prediction_mode = st.radio(
        "예측 모드 선택",
        ('인장 (ε-N)', '전단 (γ-N)'),
        key='prediction_mode'
    )
    mode_arg = 'tensile' if prediction_mode == '인장 (ε-N)' else 'shear'

    st.subheader("재료 특성")
    e_mod = st.number_input('탄성 계수 (E, MPa)', min_value=1.0, value=200000.0, format='%.1f')
    ys = st.number_input('항복 강도 (YS, MPa)', min_value=1.0, value=500.0, format='%.1f')
    ts = st.number_input('인장 강도 (UTS, MPa)', min_value=1.0, value=700.0, format='%.1f')
    hb_input = st.number_input('브리넬 경도 (HB)', min_value=0.0, value=200.0, format='%.1f', help="물리 기반 spf/epf 계산에 필요합니다.")
    poisson_ratio = st.number_input("포아송 비 (ν)", min_value=0.0, max_value=0.5, value=0.3, step=0.01, format='%.2f', help="전단 계산에 사용됩니다")

    # HB 처리
    hb_processed = hb_input
    if hb_input <= 0:
         st.warning("물리 기반 spf/epf 계산을 위해 HB 값이 필요합니다. 유효한 HB 값을 입력하세요.")
         hb_processed = 1

    predict_button = st.button(f'{prediction_mode} 곡선 예측하기')

# 메인 영역에 결과 표시
st.header("예측 결과")

if predict_button:
    # 입력값 유효성 검사
    if e_mod <= 0 or ys <= 0 or ts <= 0 or hb_processed <= 0:
        st.error("E, YS, TS, HB에 유효한 양수 값을 입력하세요.")
    elif ys > ts * 1.05:
        st.warning("항복 강도(YS)가 인장 강도(UTS)에 비해 높습니다. 값을 확인하세요.")
    else:
        try:
            # 함수 호출 및 반환값 처리
            (tensile_p, rev_en, strain_en_tot, strain_en_el, strain_en_pl,
             shear_p, rev_gn, strain_gn_tot, strain_gn_el, strain_gn_pl) = predict_fatigue_curves_hybrid(
                e_mod, ys, ts, hb_processed,
                model, scaler_X, scaler_y, device,
                nu=poisson_ratio
            )

            # 표시할 파라미터 및 그래프 데이터 선택
            if mode_arg == 'tensile':
                predicted_params = tensile_p
                reversals = rev_en
                strain_tot = strain_en_tot
                strain_el = strain_en_el
                strain_pl = strain_en_pl
                ylabel = 'Strain Amplitude (ε_a)'
                title_suffix = '(E-N)'
                tot_label = 'Total Strain (ε_a)'
                el_label = 'Elastic Strain'
                pl_label = 'Plastic Strain'
                method_caption = f"Spf/epf based on: {tensile_p.get('estimation_method_spf_epf', 'N/A')}"
                latex_formula = r"\frac{\Delta\epsilon}{2} = \frac{\sigma'_f}{E}\,(2N_f)^b + \epsilon'_f\,(2N_f)^c"

            else: # mode_arg == 'shear'
                predicted_params = shear_p
                reversals = rev_gn
                strain_tot = strain_gn_tot
                strain_el = strain_gn_el
                strain_pl = strain_gn_pl
                ylabel = 'Shear Strain Amplitude (γ_a)'
                title_suffix = '(Gamma-N)'
                tot_label = 'Total Shear Strain (γ_a)'
                el_label = 'Elastic Shear Strain'
                pl_label = 'Plastic Shear Strain'
                method_caption = f"Tauf/gammaf based on: {shear_p.get('conversion_method', 'N/A')}"
                # 보간법 추가 설명
                method = shear_p.get('conversion_method', '')
                if method.startswith("Interpolated"):
                    import re
                    m = re.search(r"α=([0-9.]+)", method)
                    alpha = float(m.group(1)) if m else None
                    method_caption += f"\n(Interpolation factor α = {alpha:.2f})"

                latex_formula = r"\frac{\Delta\gamma}{2} = \frac{\tau'_f}{G}\,(2N_f)^{b_0} + \gamma'_f\,(2N_f)^{c_0}"

            # 결과를 두 열로 표시
            col1, col2 = st.columns([1, 1])

            with col1:
                # 파라미터 표시
                st.subheader(f"예측된 {prediction_mode} 파라미터:")
                mapping = {'spf_MPa': 'σ′f (MPa)','b': 'b','epf': 'ε′f','c': 'c','tauf_MPa': 'τ′f (MPa)','gammaf': 'γ′f','b0': 'b₀','c0': 'c₀','estimation_method_spf_epf': 'Fatigue Parameter Estimation', 'conversion_method': 'Shear Conversion'}
                param_names = list(predicted_params.keys())
                param_values = [predicted_params[name] for name in param_names]
                display_names = [mapping.get(n, n) for n in param_names]
                param_df = pd.DataFrame({'Parameter': display_names,'Value': [val if isinstance(val, str) else f'{val:.4f}' for val in param_values]})
                st.dataframe(param_df)
                st.caption(method_caption)
                
                # LaTeX 공식 표시
                st.subheader("수학적 모델:")
                st.latex(latex_formula)

            with col2:
                # 그래프 표시
                st.subheader(f"예측된 {prediction_mode} 곡선:")
                fig, ax = plt.subplots(figsize=(8, 6))

                if not np.isnan(strain_tot).all():
                    ax.loglog(reversals, strain_tot, '-', label=tot_label)
                    ax.loglog(reversals, strain_el, '--', label=el_label, alpha=0.7)
                    ax.loglog(reversals, strain_pl, ':', label=pl_label, alpha=0.7)
                    ax.set_xlabel('Reversals to Failure (2Nf)')
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'Predicted {title_suffix} Curve (Components)')
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                    ax.set_ylim(bottom=max(1e-5, min(strain_tot[~np.isnan(strain_tot)])*0.5 if not np.isnan(strain_tot).all() else 1e-5),
                                top=max(strain_tot[~np.isnan(strain_tot)])*1.2 if not np.isnan(strain_tot).all() else 1e-1)
                else:
                    ax.text(0.5, 0.5, f'Could not generate {prediction_mode} Curve', ha='center', va='center')
                    ax.set_title(f'Predicted {title_suffix} Curve (Calculation Failed)')
                
                st.pyplot(fig)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
else:
    st.info('재료 특성을 입력하고, 모드를 선택한 후 "예측하기" 버튼을 누르세요.')

# 앱 실행 안내
print("\nStreamlit app code updated for hybrid parameter prediction and component plotting.")
# print(streamlit_code) # 코드 출력 (옵션)
# streamlit run FatiguePredictor.py
