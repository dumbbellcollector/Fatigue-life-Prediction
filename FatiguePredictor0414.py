# Cell 12: GUI 고려사항 (Streamlit 코드 업데이트)

# Streamlit 앱 코드 문자열 업데이트
# - 모드 선택 제거
# - predict_fatigue_curves 함수 호출
# - 인장/전단 파라미터 및 변환 방법 표시
# - E-N 및 Gamma-N 그래프 분리 표시

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd

# --- 모델 정의 복제 (FatiguePINN) ---
# (Cell 7 코드 붙여넣기)
class FatiguePINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# --- 예측 함수 복제 (predict_fatigue_curves) ---
# (Cell 11 코드 붙여넣기)
def predict_fatigue_curves(E_val, YS_val, TS_val, HB_val, model, scaler_X, scaler_y, device, mode='tensile', nu=0.3):
    model.eval()
    input_features = np.array([[E_val, YS_val, TS_val, HB_val]])
    input_scaled = scaler_X.transform(input_features)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_tensile_params_scaled = model(input_tensor)
    predicted_tensile_params_orig = scaler_y.inverse_transform(predicted_tensile_params_scaled.cpu().numpy())[0]
    tensile_target_cols_local = ['spf_MPa', 'b', 'epf', 'c']
    tensile_params = {name: val for name, val in zip(tensile_target_cols_local, predicted_tensile_params_orig)}
    spf_prime = tensile_params['spf_MPa']
    b = tensile_params['b']
    epf_prime = tensile_params['epf']
    c = tensile_params['c']
    reversals = np.logspace(1, 7, num=100)
    E_val_safe = max(E_val, 1e-6)
    elastic_strain = (spf_prime / E_val_safe) * (reversals ** b)
    plastic_strain = epf_prime * (reversals ** c)
    strain_amplitude_en = elastic_strain + plastic_strain

    shear_params = {}
    conversion_method = "Unknown"
    if TS_val <= 1100:
        tauf_prime = spf_prime / np.sqrt(3) if spf_prime is not None else None
        gammaf_prime = np.sqrt(3) * epf_prime if epf_prime is not None else None
        b0 = b; c0 = c; conversion_method = "von Mises Criterion"
    elif TS_val >= 1696:
        tauf_prime = spf_prime / np.sqrt(3) if spf_prime is not None else None
        gammaf_prime = np.sqrt(3) * epf_prime if epf_prime is not None else None
        b0 = b; c0 = c; conversion_method = "Maximum Principal Criterion"
    else:
        tauf_prime = spf_prime / np.sqrt(3) if spf_prime is not None else None
        gammaf_prime = np.sqrt(3) * epf_prime if epf_prime is not None else None
        b0 = b; c0 = c; conversion_method = "von Mises (Intermediate Default)"

    shear_params['tauf_MPa'] = tauf_prime if tauf_prime is not None else np.nan
    shear_params['gammaf'] = gammaf_prime if gammaf_prime is not None else np.nan
    shear_params['b0'] = b0 if b0 is not None else np.nan
    shear_params['c0'] = c0 if c0 is not None else np.nan
    shear_params['conversion_method'] = conversion_method

    G_val = E_val_safe / (2 * (1 + nu))
    elastic_shear_strain = np.full_like(reversals, np.nan)
    plastic_shear_strain = np.full_like(reversals, np.nan)
    strain_amplitude_gn = np.full_like(reversals, np.nan)
    
    if not any(np.isnan(val) for key, val in shear_params.items() if key != 'conversion_method'):
         elastic_shear_strain = (tauf_prime / G_val) * (reversals ** b0)
         plastic_shear_strain = gammaf_prime * (reversals ** c0)
         strain_amplitude_gn = elastic_shear_strain + plastic_shear_strain
    else:
         st.warning("Could not calculate shear curve due to invalid converted parameters.") # GUI 경고
    
    if mode == 'tensile':
        return tensile_params, reversals, strain_amplitude_en, elastic_strain, plastic_strain
    else:  # mode == 'shear'
        return shear_params, reversals, strain_amplitude_gn, elastic_shear_strain, plastic_shear_strain

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
        model = FatiguePINN(input_dim, output_dim_tensile).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, scaler_X, scaler_y, device
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('Fatigue Life Predictor (E-N / Gamma-N)')
st.write("Enter material properties, select mode, and predict.")

# 리소스 로드
model, scaler_X, scaler_y, device = load_resources()

if model is None: st.stop()

# 입력 섹션
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    # --- 예측 모드 선택 다시 추가 ---
    prediction_mode = st.radio(
        "Select Prediction Mode",
        ('Tensile (E-N)', 'Shear (Gamma-N)'),
        key='prediction_mode'
    )
    mode_arg = 'tensile' if prediction_mode == 'Tensile (E-N)' else 'shear'

    st.subheader("Material Properties")
    e_mod = st.number_input('Elastic Modulus (E, MPa)', min_value=1.0, value=200000.0, format='%.1f')
    ys = st.number_input('Yield Strength (YS, MPa)', min_value=1.0, value=500.0, format='%.1f')
    ts = st.number_input('Ultimate Tensile Strength (UTS, MPa)', min_value=1.0, value=700.0, format='%.1f')
    hb_input = st.number_input('Brinell Hardness (HB)', min_value=0.0, value=200.0, format='%.1f', help="Enter 0 if unknown.")
    poisson_ratio = st.number_input("Poisson's Ratio (nu)", min_value=0.0, max_value=0.5, value=0.3, step=0.01, format='%.2f', help="Used for Shear calculation")

    # HB 처리 (간단히 평균값 사용 예시 - 실제로는 HV 변환 로직 필요 또는 사용자 입력 강제)
    hb_processed = hb_input
    if hb_input == 0.0:
         try:
            hb_mean_from_scaler = scaler_X.mean_[-1] # 마지막 특성이 HB라고 가정
            hb_processed = hb_mean_from_scaler
            st.info(f"HB not provided. Using average HB from training data: {hb_processed:.1f}")
         except Exception:
             st.warning("Could not retrieve mean HB. Using input value 0. Prediction might be inaccurate if HB is required.")
             hb_processed = 0.1

    predict_button = st.button(f'Predict {prediction_mode} Curve')

# 결과 섹션
with col2:
    st.header("Prediction Results")
    if predict_button:
        if e_mod <= 0 or ys <= 0 or ts <= 0 or hb_processed <= 0: # HB도 유효성 검사
            st.error("Please enter valid positive values for all properties.")
        elif ys > ts * 1.05:
            st.warning("Yield Strength (YS) seems high compared to UTS. Please verify.")
        else:
            try:
                # --- 함수 호출 및 반환값 업데이트 ---
                predicted_params, reversals, strain_tot, strain_el, strain_pl = predict_fatigue_curves(
                    e_mod, ys, ts, hb_processed,
                    model, scaler_X, scaler_y, device,
                    mode=mode_arg,
                    nu=poisson_ratio
                )

                # --- 파라미터 표시 (모드별) ---
                st.subheader(f"Predicted {prediction_mode} Parameters:")
                param_names = list(predicted_params.keys())
                param_values = [predicted_params[name] for name in param_names]
                param_df = pd.DataFrame({'Parameter': param_names,
                                         'Value': [val if isinstance(val, str) else f'{val:.4f}' for val in param_values]})
                st.dataframe(param_df)

                # --- 그래프 표시 (3개 요소 함께) ---
                st.subheader(f"Predicted {prediction_mode} Curve:")
                fig, ax = plt.subplots(figsize=(8, 6))

                ylabel = 'Strain Amplitude (epsilon_a)' if mode_arg == 'tensile' else 'Shear Strain Amplitude (gamma_a)'
                title = f'Predicted {prediction_mode} Curve (Components)'
                tot_label = 'Total Strain (epsilon_a)' if mode_arg == 'tensile' else 'Total Shear Strain (gamma_a)'
                el_label = 'Elastic Strain' if mode_arg == 'tensile' else 'Elastic Shear Strain'
                pl_label = 'Plastic Strain' if mode_arg == 'tensile' else 'Plastic Shear Strain'

                if not np.isnan(strain_tot).all(): # NaN 이 아닌 경우만 플롯
                    ax.loglog(reversals, strain_tot, '-', label=tot_label)
                    ax.loglog(reversals, strain_el, '--', label=el_label, alpha=0.7)
                    ax.loglog(reversals, strain_pl, ':', label=pl_label, alpha=0.7)
                    ax.set_xlabel('Reversals to Failure (2Nf)')
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                    ax.set_ylim(bottom=max(1e-5, min(strain_tot)*0.5), top=max(strain_tot)*1.2)
                else:
                    ax.text(0.5, 0.5, f'Could not generate {prediction_mode} Curve', ha='center', va='center')
                    ax.set_title(f'Predicted {prediction_mode} Curve (Calculation Failed)')

                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                # import traceback
                # st.text(traceback.format_exc()) # 디버깅용

    else:
        st.info('Enter properties, select mode, and click "Predict".')


# 앱 실행 안내
print("\nStreamlit app code updated to re-enable mode selection and plot strain components.")
# print(streamlit_code) # 코드 출력 (옵션)
# streamlit run app0414.py
