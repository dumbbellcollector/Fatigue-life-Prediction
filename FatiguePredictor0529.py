import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import io # For image download

# Import the new module
import composition_to_properties as ctp

# --- File Paths (Global Constants) ---
# Attempt to determine script directory for robust pathing
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: # Occurs when running in some environments like notebooks directly
    BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, 'best_fatigue_pinn_model.pth')
SCALER_X_PATH = os.path.join(BASE_DIR, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(BASE_DIR, 'scalers_y.pkl')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Streamlit App: Using device: {device}")

# --- Input Validation Function ---
def validate_monotonic_inputs(e_mod_gpa, ys_mpa, ts_mpa, hb_input, poisson_ratio):
    error_messages = []
    if e_mod_gpa is None or e_mod_gpa <= 0: error_messages.append("탄성 계수(E)는 0보다 커야 합니다 (GPa 단위).")
    if ys_mpa is None or ys_mpa <= 0: error_messages.append("항복 강도(YS)는 0보다 커야 합니다 (MPa 단위).")
    if ts_mpa is None or ts_mpa <= 0: error_messages.append("인장 강도(TS)는 0보다 커야 합니다 (MPa 단위).")
    if hb_input is not None and hb_input < 0: error_messages.append("브리넬 경도(HB)는 음수일 수 없습니다.")
    if poisson_ratio is None or not (0 <= poisson_ratio <= 0.5): error_messages.append("포아송 비(ν)는 0.0에서 0.5 사이여야 합니다.")
    
    if ys_mpa is not None and ts_mpa is not None and ys_mpa > ts_mpa and ys_mpa > 0 and ts_mpa > 0 :
        st.sidebar.warning("경고: 항복 강도(YS)가 인장 강도(TS)보다 큽니다. 입력값을 확인하세요.")

    if error_messages:
        for msg in error_messages:
            st.sidebar.error(msg)
        return False
    return True

def validate_composition_inputs(composition: dict):
    error_messages = []
    for element, value in composition.items():
        if value is None or value < 0: # Also check for None
            error_messages.append(f"{element}의 조성비는 0 이상이어야 합니다.")
    
    # Optional: Check sum, but usually Fe is balance
    # total_comp = sum(v for v in composition.values() if v is not None)
    # if total_comp > 100:
    #     error_messages.append(f"입력된 조성의 합계({total_comp:.2f} wt%)가 100 wt%를 초과합니다.")

    if error_messages:
        for msg in error_messages:
            st.sidebar.error(msg)
        return False
    return True

# --- main.ipynb의 inverse_transform_targets 함수 ---
def inverse_transform_targets(y_scaled_data, scalers_y_dict, target_cols_list):
    if y_scaled_data.ndim == 1:
        y_scaled_data = y_scaled_data.reshape(1, -1)
        
    y_transformed_individually = np.zeros_like(y_scaled_data)
    for i, col_name in enumerate(target_cols_list):
        if col_name not in scalers_y_dict: 
            raise KeyError(f"Scaler for target '{col_name}' not found in scalers_y_dict. Available keys: {list(scalers_y_dict.keys())}")
        y_transformed_individually[:, i] = scalers_y_dict[col_name].inverse_transform(y_scaled_data[:, i].reshape(-1, 1)).flatten()
    
    y_orig_scale = y_transformed_individually.copy()
    
    log_col_expected_name = 'epf' 
    fallback_log_col_name = 'epf_log'

    if log_col_expected_name in target_cols_list:
        try:
            current_log_col_idx = target_cols_list.index(log_col_expected_name)
            y_orig_scale[:, current_log_col_idx] = np.expm1(y_transformed_individually[:, current_log_col_idx])
        except ValueError:
            st.warning(f"'{log_col_expected_name}' was in target_cols_list but index could not be found. Skipping expm1 transformation for it.")
    elif fallback_log_col_name in target_cols_list: 
        try:
            current_fallback_log_col_idx = target_cols_list.index(fallback_log_col_name)
            st.info(f"Note: Found '{fallback_log_col_name}' in target_cols and applying expm1. "
                    f"The primary expected log-column name is '{log_col_expected_name}'.")
            y_orig_scale[:, current_fallback_log_col_idx] = np.expm1(y_transformed_individually[:, current_fallback_log_col_idx])
        except ValueError:
            st.warning(f"'{fallback_log_col_name}' was in target_cols_list but index could not be found. Skipping expm1 transformation for it.")
            
    if y_scaled_data.shape[0] == 1 and y_orig_scale.shape[0] == 1:
         return y_orig_scale.flatten()
    return y_orig_scale

# --- 모델 정의 복제 (FatiguePINN) ---
class FatiguePINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128], dropout_p=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p 

        layers = []
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
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
    spf = 1.9 * ts_mpa
    epf = 0.623 * (ts_mpa / e_mpa) ** 0.832
    return spf, epf

def uniform_material_law(ts_mpa, e_mpa):
    if ts_mpa < 750:
        spf = 1.5 * ts_mpa
        epf = 0.59 * (ts_mpa / e_mpa) ** 0.58
    else:
        spf = 1.67 * ts_mpa
        epf = 0.35
    return spf, epf

def hardness_method(hb, ts_mpa, e_mpa):
    spf = 4.25 * hb + 225
    epf = 0.32 * (hb / 1000) ** (-1.24)
    return spf, epf

method_map = {1: hardness_method, 2: hardness_method, 3: hardness_method, 4: hardness_method}
method_names = {1: "Hardness Method", 2: "Hardness Method", 3: "Hardness Method", 4: "Hardness Method"}

def get_physics_params(hb_val, ts_mpa_val, e_mpa_val):
    method_name = "Unknown"
    spf_physics = epf_physics = None
    
    if hb_val is None or hb_val <= 0 or np.isnan(hb_val):
        return np.nan, np.nan, "유효한 HB 값이 없어 물리 기반 계산 불가"
    
    if ts_mpa_val is None or e_mpa_val is None or ts_mpa_val <=0 or e_mpa_val <=0: # Added check for TS, E
        return np.nan, np.nan, "TS 또는 E 값이 유효하지 않아 물리 기반 계산 불가"

    if hb_val <= HB_CLASS1_MAX: material_class = 1
    elif hb_val <= HB_CLASS2_MAX: material_class = 2
    elif hb_val <= HB_CLASS3_MAX: material_class = 3
    else: material_class = 4
    
    method_func = method_map.get(material_class)
    method_name = method_names.get(material_class, "Unknown")
    
    if method_func:
        try:
            spf_physics, epf_physics = method_func(hb_val, ts_mpa_val, e_mpa_val)
        except Exception as e:
            return np.nan, np.nan, f"{method_name} 계산 중 오류: {str(e)}"
    
    return spf_physics, epf_physics, method_name

# --- Helper function for Shear Conversion (New) ---
def convert_to_shear_parameters(spf_prime, b_fatigue, epf_prime, c_fatigue, E_gpa, TS_mpa, nu=0.3):
    # Ensure all inputs are valid numbers before proceeding
    if any(val is None or np.isnan(val) for val in [spf_prime, b_fatigue, epf_prime, c_fatigue, E_gpa, TS_mpa, nu]):
        st.warning("전단 변환에 필요한 일부 파라미터가 유효하지 않습니다. (NaN 또는 None)")
        return { # Return a structure with NaNs to prevent downstream errors
            'tauf_MPa': np.nan, 'gammaf': np.nan, 'b0': np.nan, 'c0': np.nan,
            'conversion_method': "입력 파라미터 부족/무효", 'G_mpa': np.nan
        }

    tau_vm = spf_prime / np.sqrt(3)
    gamma_vm = np.sqrt(3) * epf_prime
    
    tau_mp = spf_prime / (1 + nu) 
    gamma_mp = 2 * epf_prime

    b0 = b_fatigue 
    c0 = c_fatigue

    conversion_method = "Unknown"
    tauf_prime_calc, gammaf_prime_calc = np.nan, np.nan # Renamed to avoid conflict

    if TS_mpa <= 1100:
        tauf_prime_calc, gammaf_prime_calc = tau_vm, gamma_vm
        conversion_method = "von Mises Criteria"
    elif TS_mpa >= 1696:
        tauf_prime_calc, gammaf_prime_calc = tau_mp, gamma_mp
        conversion_method = "Maximum Principal Stress/Strain Criteria"
    else:
        alpha = (TS_mpa - 1100) / (1696 - 1100)
        tauf_prime_calc = (1 - alpha) * tau_vm + alpha * tau_mp
        gammaf_prime_calc = (1 - alpha) * gamma_vm + alpha * gamma_mp
        conversion_method = f"Interpolated (von Mises to Max Principal, α={alpha:.2f})"

    E_mpa_internal = E_gpa * 1000
    G_mpa_internal = E_mpa_internal / (2 * (1 + nu)) if E_mpa_internal > 0 and nu is not None else np.nan

    shear_params = {
        'tauf_MPa': tauf_prime_calc,
        'gammaf': gammaf_prime_calc,
        'b0': b0,
        'c0': c0,
        'conversion_method': conversion_method,
        'G_mpa': G_mpa_internal
    }
    return shear_params

# --- Hybrid Model Prediction and Curve Generation (Modified) ---
def predict_fatigue_curves_hybrid(E_val_gpa, YS_val_mpa, TS_val_mpa, HB_val, model, scaler_X, 
                                  scalers_y_dict, target_cols_list,
                                  device): 
    
    if any(v is None for v in [E_val_gpa, YS_val_mpa, TS_val_mpa]): # HB_val can be None
        raise ValueError("E, YS, TS 값은 반드시 제공되어야 합니다.")

    E_val_mpa = E_val_gpa * 1000 

    hb_processed_val = HB_val
    if HB_val is None or np.isnan(HB_val) or HB_val <= 0: # Added HB_val <= 0
        if TS_val_mpa is not None and not np.isnan(TS_val_mpa) and TS_val_mpa > 0:
            hb_processed_val = 1.8 * TS_val_mpa + 105 
            st.sidebar.info(f"HB값이 제공되지 않거나 유효하지 않아 TS ({TS_val_mpa} MPa)로부터 추정된 HB ({hb_processed_val:.1f})를 사용합니다.")
        else: 
             raise ValueError("HB 값과 TS_MPa 값 모두 유효하지 않아 HB_processed를 계산할 수 없습니다.")
    
    # Ensure all features for model are valid numbers
    if any(np.isnan(v) or v is None for v in [E_val_mpa, YS_val_mpa, TS_val_mpa, hb_processed_val]):
        raise ValueError(f"모델 입력 특징 중 유효하지 않은 값이 있습니다: E={E_val_mpa}, YS={YS_val_mpa}, TS={TS_val_mpa}, HB_proc={hb_processed_val}")


    features_np = np.array([[E_val_mpa, YS_val_mpa, TS_val_mpa, hb_processed_val]], dtype=np.float32)
    
    scaled_features_np = scaler_X.transform(features_np)
    scaled_features = torch.tensor(scaled_features_np, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predicted_params_scaled = model(scaled_features)
        
    predicted_params_orig_array = inverse_transform_targets(
        predicted_params_scaled.cpu().numpy(), 
        scalers_y_dict, 
        target_cols_list
    )
    
    predicted_params_orig = predicted_params_orig_array

    params_dict = dict(zip(target_cols_list, predicted_params_orig))

    sigma_f_prime = params_dict.get('spf_MPa')
    b_fatigue = params_dict.get('b')

    if 'epf' in params_dict:
        epsilon_f_prime = params_dict.get('epf')
    elif 'epf_log' in params_dict:
        epsilon_f_prime = params_dict.get('epf_log')
    else:
        epsilon_f_prime = None 

    c_fatigue = params_dict.get('c')

    if sigma_f_prime is None or b_fatigue is None or epsilon_f_prime is None or c_fatigue is None:
        epf_key_in_list = 'epf' if 'epf' in target_cols_list else 'epf_log' if 'epf_log' in target_cols_list else 'epf/epf_log'
        missing_keys = [key for key, val in zip(['spf_MPa', 'b', epf_key_in_list, 'c'], 
                                               [sigma_f_prime, b_fatigue, epsilon_f_prime, c_fatigue]) if val is None]
        raise ValueError(f"모델 예측에서 다음 필수 파라미터들을 얻을 수 없습니다: {missing_keys}. "
                         f"사용된 target_cols_list: {target_cols_list}, params_dict에 있는 키: {list(params_dict.keys())}")

    Nf = np.logspace(1, 7, 100) 
    delta_epsilon_el_half = (sigma_f_prime / E_val_mpa) * (2 * Nf)**b_fatigue
    delta_epsilon_pl_half = epsilon_f_prime * (2 * Nf)**c_fatigue
    delta_epsilon_tot_half = delta_epsilon_el_half + delta_epsilon_pl_half
    sigma_a = sigma_f_prime * (2 * Nf)**b_fatigue

    results = {
        "Nf": Nf,
        "total_strain_amplitude": delta_epsilon_tot_half,
        "elastic_strain_amplitude": delta_epsilon_el_half,
        "plastic_strain_amplitude": delta_epsilon_pl_half,
        "stress_amplitude": sigma_a,
        "sigma_f_prime": sigma_f_prime,
        "b_fatigue": b_fatigue,
        "epsilon_f_prime": epsilon_f_prime,
        "c_fatigue": c_fatigue,
        "E_mpa": E_val_mpa, 
        "HB_processed_for_prediction": hb_processed_val
    }
    return results

# --- Load Model and Scalers ---
@st.cache_resource
def load_resources(model_path=MODEL_PATH, scaler_x_path=SCALER_X_PATH, scaler_y_path=SCALER_Y_PATH):
    model_hidden_dims = [192, 384, 352, 224]
    model_dropout_p = 0.35
    input_dim = 4 
    output_dim = 4 

    model = FatiguePINN(input_dim=input_dim, output_dim=output_dim, 
                        hidden_dims=model_hidden_dims, dropout_p=model_dropout_p)
    
    _device = device 

    if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_x_path): raise FileNotFoundError(f"Scaler X file not found: {scaler_x_path}")
    if not os.path.exists(scaler_y_path): raise FileNotFoundError(f"Scaler Y file not found: {scaler_y_path}")

    map_location = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.to(_device)
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    data_y = joblib.load(scaler_y_path)
    
    expected_cols_for_model = ['spf_MPa', 'b', 'epf', 'c'] 

    if isinstance(data_y, dict) and 'scalers' in data_y and 'target_cols' in data_y:
        scalers_y_dict = data_y['scalers']
        target_cols_list = data_y['target_cols']
        
        if set(target_cols_list) != set(expected_cols_for_model):
            if 'epf_log' in target_cols_list and 'epf' not in target_cols_list and 'epf' in expected_cols_for_model:
                st.warning(
                    f"경고: 로드된 target_cols ({target_cols_list})에 'epf_log'가 포함되어 있습니다. "
                    f"앱은 '{expected_cols_for_model}' ('epf')를 기대합니다. 'epf_log'를 대체 처리합니다."
                )
            else:
                 st.warning(
                    f"경고: 로드된 target_cols ({target_cols_list})가 예상 기본값 ({expected_cols_for_model})과 다릅니다. "
                    f"로드된 값을 사용하나, 예상치 못한 동작이 발생할 수 있습니다."
                )
        if target_cols_list != expected_cols_for_model and set(target_cols_list) == set(expected_cols_for_model):
            st.info(f"로드된 target_cols ({target_cols_list})의 순서가 예상된 순서 ({expected_cols_for_model})와 다릅니다. 이름은 일치하므로 계속 진행합니다.")

        missing_scalers = [col for col in target_cols_list if col not in scalers_y_dict]
        if missing_scalers:
            raise ValueError(f"'{scaler_y_path}'의 'scalers'에 다음 타겟 컬럼 스케일러가 없습니다: {missing_scalers}.")

    elif isinstance(data_y, dict):
        st.warning(f"경고: '{scaler_y_path}' 파일이 이전 형식입니다. 기본 타겟 컬럼 {expected_cols_for_model}을 사용합니다.")
        scalers_y_dict = data_y 
        target_cols_list = expected_cols_for_model 
        missing_keys_in_scaler = [key for key in target_cols_list if key not in scalers_y_dict]
        if missing_keys_in_scaler:
            raise ValueError(f"이전 형식 '{scaler_y_path}' 파일의 스케일러에 누락된 키: {missing_keys_in_scaler}")
    else:
        raise ValueError(f"'{scaler_y_path}' 파일 형식을 인식할 수 없습니다.")

    if model.output_dim != len(target_cols_list):
        st.error(f"모델 출력 차원({model.output_dim})과 타겟 컬럼 수({len(target_cols_list)}) 불일치. 타겟: {target_cols_list}")
        raise ValueError("Model output dimension and target columns mismatch.")
        
    return model, scaler_X, scalers_y_dict, target_cols_list

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Fatigue Life Predictor")
st.title('Fatigue Life Predictor (ε-N / γ-N)')
st.write("Enter Monotonic Properties or Alloy Composition, Get Prediction.")

# --- Session State 초기화 ---
if 'prediction_triggered' not in st.session_state: st.session_state.prediction_triggered = False
if 'user_inputs' not in st.session_state: st.session_state.user_inputs = {}
if 'en_results' not in st.session_state: st.session_state.en_results = None
if 'shear_results' not in st.session_state: st.session_state.shear_results = None
if 'physics_params' not in st.session_state: st.session_state.physics_params = None
if 'input_mode' not in st.session_state: st.session_state.input_mode = "단조 물성치 직접 입력"
if 'poisson_ratio_universal' not in st.session_state: st.session_state.poisson_ratio_universal = 0.3 # 공용 포아송 비 초기화

# 리소스 로드 (앱 시작 시 한 번)
try:
    model, scaler_X, scalers_y_dict, target_cols_list = load_resources()
except FileNotFoundError as e:
    st.error(f"필수 파일 로드 실패: {e}. 앱을 실행할 수 없습니다.")
    st.stop()
except ValueError as e:
    st.error(f"데이터 파일 형식 또는 내용 오류: {e}. 앱을 실행할 수 없습니다.")
    st.stop()
except Exception as e:
    st.error(f"알 수 없는 오류로 리소스 로드 실패: {e}")
    st.stop()


# Define elements for composition input globally for access
elements_for_input_definition = {
    'C': {'label': 'C (탄소)', 'default': 0.2, 'format': '%.3f', 'step': 0.001},
    'Mn': {'label': 'Mn (망간)', 'default': 0.8, 'format': '%.3f', 'step': 0.01},
    'Si': {'label': 'Si (규소)', 'default': 0.3, 'format': '%.3f', 'step': 0.01},
    'Cr': {'label': 'Cr (크롬)', 'default': 0.0, 'format': '%.3f', 'step': 0.01},
    'Mo': {'label': 'Mo (몰리브덴)', 'default': 0.0, 'format': '%.3f', 'step': 0.01},
    'Ni': {'label': 'Ni (니켈)', 'default': 0.0, 'format': '%.3f', 'step': 0.01},
    'V': {'label': 'V (바나듐)', 'default': 0.0, 'format': '%.3f', 'step': 0.001},
    'Nb': {'label': 'Nb (니오븀)', 'default': 0.0, 'format': '%.4f', 'step': 0.0001},
    'Ti': {'label': 'Ti (티타늄)', 'default': 0.0, 'format': '%.4f', 'step': 0.0001},
    'Al': {'label': 'Al (알루미늄)', 'default': 0.0, 'format': '%.3f', 'step': 0.001},
    'N': {'label': 'N (질소)', 'default': 0.0, 'format': '%.4f', 'step': 0.0001},
    'Cu': {'label': 'Cu (구리)', 'default': 0.0, 'format': '%.3f', 'step': 0.01},
    'P': {'label': 'P (인)', 'default': 0.015, 'format': '%.4f', 'step': 0.001},
    'S': {'label': 'S (황)', 'default': 0.015, 'format': '%.4f', 'step': 0.001},
    'B': {'label': 'B (붕소)', 'default': 0.0, 'format': '%.5f', 'step': 0.00001}
}

# 사이드바에 입력 섹션 배치
with st.sidebar:
    st.header("입력 모드 및 파라미터")

    st.radio(
        "입력 방식 선택:",
        ("단조 물성치 직접 입력", "합금 조성비 입력 (wt%)"),
        key='input_mode',
        horizontal=True,
        on_change=lambda: setattr(st.session_state, 'prediction_triggered', False)
    )

    if st.session_state.input_mode == "단조 물성치 직접 입력":
        st.subheader("재료 특성 (직접 입력)")
        e_mod_gpa_direct_input = st.number_input('탄성 계수 (E, GPa)', min_value=1.0, value=200.0, format='%.1f', help="GPa 단위. (예: 강철 ~200 GPa)", key="e_mod_gpa_direct")
        ys_mpa_direct_input = st.number_input('항복 강도 (YS, MPa)', min_value=1.0, value=500.0, format='%.1f', key="ys_mpa_direct")
        ts_mpa_direct_input = st.number_input('인장 강도 (UTS, MPa)', min_value=1.0, value=700.0, format='%.1f', key="ts_mpa_direct")
        hb_direct_input_val = st.number_input('브리넬 경도 (HB)', min_value=0.0, value=200.0, format='%.1f', help="물리 기반 계산 및 모델 입력용. 0 입력 시 TS로부터 추정.", key="hb_direct")
        poisson_direct_input = st.number_input("포아송 비 (ν)", min_value=0.0, max_value=0.5, value=0.3, step=0.01, format='%.2f', help="전단 계산에 사용.", key="poisson_direct")

        if hb_direct_input_val is not None and hb_direct_input_val <= 0:
             st.info("HB 값이 0 이하로 입력되어 인장강도(TS)로부터 HB를 추정하여 사용합니다 (모델 입력 및 물리식 계산 시).")

    elif st.session_state.input_mode == "합금 조성비 입력 (wt%)":
        st.subheader("합금 조성 (wt%)")
        
        col1_comp, col2_comp = st.columns(2)
        element_keys_col1 = list(elements_for_input_definition.keys())[:8]
        
        for el_key, props in elements_for_input_definition.items():
            target_col = col1_comp if el_key in element_keys_col1 else col2_comp
            with target_col:
                st.number_input(
                    props['label'], min_value=0.0, value=props['default'], 
                    format=props['format'], step=props['step'], key=f"comp_{el_key}"
                )
        
        current_composition_sum = sum(st.session_state[f"comp_{el}"] for el in elements_for_input_definition if f"comp_{el}" in st.session_state)
        st.caption(f"입력된 조성의 합계: {current_composition_sum:.3f} wt%. (Fe는 잔량으로 간주)")

    if st.button("피로 거동 예측 실행", use_container_width=True, type="primary"):
        st.session_state.prediction_triggered = True
        st.session_state.en_results = None
        st.session_state.shear_results = None
        st.session_state.physics_params = None
        st.session_state.user_inputs = {}

        # Initialize common variables for prediction
        e_gpa_to_use, ys_mpa_to_use, ts_mpa_to_use, hb_to_use, nu_to_use = None, None, None, None, None
        input_mode_str = st.session_state.input_mode
        
        try:
            if st.session_state.input_mode == "단조 물성치 직접 입력":
                e_gpa_to_use = st.session_state.e_mod_gpa_direct
                ys_mpa_to_use = st.session_state.ys_mpa_direct
                ts_mpa_to_use = st.session_state.ts_mpa_direct
                hb_to_use = st.session_state.hb_direct
                nu_to_use = st.session_state.poisson_direct
                
                if not validate_monotonic_inputs(e_gpa_to_use, ys_mpa_to_use, ts_mpa_to_use, hb_to_use, nu_to_use):
                    raise ValueError("직접 입력된 재료 물성치 유효성 검사 실패.")
                
                st.session_state.user_inputs = {
                    'E_gpa': e_gpa_to_use, 'YS_mpa': ys_mpa_to_use, 'TS_mpa': ts_mpa_to_use,
                    'HB': hb_to_use, 'nu': nu_to_use, 'Input_Mode': input_mode_str
                }

            elif st.session_state.input_mode == "합금 조성비 입력 (wt%)":
                current_composition = {el: st.session_state[f"comp_{el}"] for el in elements_for_input_definition}
                
                if not validate_composition_inputs(current_composition):
                    raise ValueError("입력된 합금 조성 유효성 검사 실패.")

                # e_gpa_to_use는 예측값을 사용하고, nu_to_use는 공용 값을 사용
                predicted_props = ctp.calculate_monotonic_properties(current_composition)
                e_gpa_to_use = predicted_props.get('E_gpa') 
                ys_mpa_to_use = predicted_props.get('YS_mpa')
                ts_mpa_to_use = predicted_props.get('TS_mpa')
                hb_to_use = predicted_props.get('HB') # Can be None or NaN
                nu_to_use = st.session_state.poisson_ratio_universal # 공용 포아송 비 사용

                if not validate_monotonic_inputs(e_gpa_to_use, ys_mpa_to_use, ts_mpa_to_use, hb_to_use, nu_to_use):
                    raise ValueError("조성으로부터 예측/입력된 물성치 유효성 검사 실패.")
                
                st.session_state.user_inputs = {
                    'E_gpa': e_gpa_to_use, 'YS_mpa': ys_mpa_to_use, 'TS_mpa': ts_mpa_to_use,
                    'HB': hb_to_use, 'nu': nu_to_use, 'Input_Mode': input_mode_str,
                    'Composition_Details': current_composition
                }
                st.sidebar.success("조성 기반 물성치 예측 완료. 피로 거동 분석 진행.")

            # Common prediction logic using *_to_use variables
            e_mod_mpa_calc = e_gpa_to_use * 1000
            st.session_state.user_inputs['E_mpa'] = e_mod_mpa_calc # Add E_mpa to user_inputs

            # hb_to_pass handles None, NaN, or <=0 values for HB_val before model prediction
            hb_for_model = hb_to_use 
            if hb_to_use is None or np.isnan(hb_to_use) or hb_to_use <= 0:
                 hb_for_model = np.nan # predict_fatigue_curves_hybrid will estimate if TS is valid

            predicted_en_results = predict_fatigue_curves_hybrid(
                e_gpa_to_use, ys_mpa_to_use, ts_mpa_to_use, hb_for_model,
                model, scaler_X, scalers_y_dict, target_cols_list, device
            )
            st.session_state.en_results = predicted_en_results
            
            # Use hb_to_use for physics params (get_physics_params handles None/NaN/0 internally)
            spf_phys, epf_phys, phys_method_name = get_physics_params(
                hb_to_use, ts_mpa_to_use, e_mod_mpa_calc
            )
            st.session_state.physics_params = {
                'spf_MPa_phys': spf_phys, 'epf_phys': epf_phys, 'method_name': phys_method_name
            }

            st.session_state.shear_results = convert_to_shear_parameters(
                spf_prime=predicted_en_results['sigma_f_prime'],
                b_fatigue=predicted_en_results['b_fatigue'],
                epf_prime=predicted_en_results['epsilon_f_prime'],
                c_fatigue=predicted_en_results['c_fatigue'],
                E_gpa=e_gpa_to_use, TS_mpa=ts_mpa_to_use, nu=nu_to_use
            )

        except ValueError as ve: # Catches validation errors and others
            st.sidebar.error(f"오류: {ve}")
        except AttributeError as ae: # ctp module related
            st.sidebar.error(f"조성 예측 모듈 오류: {ae}. 'composition_to_properties.py'를 확인하세요.")
        except KeyError as ke: # ctp result dictionary key error
            st.sidebar.error(f"조성 예측 결과 키 오류: {ke}. 'composition_to_properties.py' 반환 형식을 확인하세요.")
        except Exception as e:
            st.sidebar.error(f"예측 중 예기치 않은 오류 발생: {e}")


# --- Main Area for Results ---
if st.session_state.prediction_triggered and st.session_state.en_results:
    tab_titles = ["피로 파라미터 요약", "인장 곡선 (ϵ-N)", "전단 곡선 (γ-N)"]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # Parameter Summary Tab
        st.subheader("피로 파라미터 요약")
        en_p = st.session_state.en_results
        phys_p = st.session_state.physics_params
        shear_p = st.session_state.shear_results
        user_p = st.session_state.user_inputs

        st.write(f"**입력 모드:** {user_p.get('Input_Mode', 'N/A')}")
        if user_p.get('Input_Mode') == "합금 조성비 입력 (wt%)":
            with st.expander("입력된 합금 조성 상세"):
                comp_details_df = pd.DataFrame(list(user_p.get('Composition_Details', {}).items()), columns=['Element', 'wt%'])
                st.dataframe(comp_details_df)
        
        st.markdown("#### AI 모델 예측 (인장)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(r"$\sigma_f'$ (피로 강도 계수, Fatigue Strength Coeff.)", f"{en_p.get('sigma_f_prime', np.nan):.2f} MPa")
            st.metric(r"$b$ (피로 강도 지수, Fatigue Strength Exponent)", f"{en_p.get('b_fatigue', np.nan):.4f}")
        with col2:
            st.metric(r"$\epsilon_f'$ (피로 연성 계수, Fatigue Ductility Coeff.)", f"{en_p.get('epsilon_f_prime', np.nan):.4f}")
            st.metric(r"$c$ (피로 연성 지수, Fatigue Ductility Exponent)", f"{en_p.get('c_fatigue', np.nan):.4f}")
        
        if phys_p and not (np.isnan(phys_p.get('spf_MPa_phys', np.nan)) and np.isnan(phys_p.get('epf_phys', np.nan))):
            st.markdown(f"#### 물리 기반 경험식 ({phys_p.get('method_name', 'N/A')})")
            col3, col4 = st.columns(2)
            with col3:
                st.metric(r"$\sigma_f'$ (계산값)", f"{phys_p.get('spf_MPa_phys', np.nan):.2f} MPa")
            with col4:
                st.metric(r"$\epsilon_f'$ (계산값)", f"{phys_p.get('epf_phys', np.nan):.4f}")

        if shear_p and not np.isnan(shear_p.get('tauf_MPa', np.nan)):
            st.markdown(f"#### 전단 변환 결과 ({shear_p.get('conversion_method', 'N/A')})")
            col5, col6 = st.columns(2)
            with col5:
                st.metric(r"$\tau_f'$ (전단 피로 강도 계수, Shear Fatigue Strength Coeff.)", f"{shear_p.get('tauf_MPa', np.nan):.2f} MPa")
                st.metric(r"$\gamma_f'$ (전단 피로 연성 계수, Shear Fatigue Ductility Coeff.)", f"{shear_p.get('gammaf', np.nan):.4f}")
                st.metric("$G$ (전단 탄성 계수, Shear Modulus)", f"{shear_p.get('G_mpa', np.nan):.0f} MPa")
            with col6: # b0 and c0 are same as b and c
                st.metric("$b_0$ (전단 피로 강도 지수, Shear Fatigue Strength Exponent)", f"{shear_p.get('b0', np.nan):.4f}")
                st.metric("$c_0$ (전단 피로 연성 지수, Shear Fatigue Ductility Exponent)", f"{shear_p.get('c0', np.nan):.4f}")
        
        st.markdown("---")
        st.markdown(r"**인장 Coffin-Manson:** $\frac{\Delta\epsilon}{2} = \frac{\sigma'_f}{E}\,(2N_f)^b + \epsilon'_f\,(2N_f)^c$")
        if shear_p and not np.isnan(shear_p.get('tauf_MPa', np.nan)):
             st.markdown(r"**전단 Coffin-Manson:** $\frac{\Delta\gamma}{2} = \frac{\tau'_f}{G}\,(2N_f)^{b_0} + \gamma'_f\,(2N_f)^{c_0}$")
        
        # Consolidate all parameters for download
        all_params_list = [
            ("Input Mode", user_p.get('Input_Mode', 'N/A')),
            ("E (GPa, Input)", user_p.get('E_gpa', np.nan)),
            ("YS (MPa, Input/Predicted)", user_p.get('YS_mpa', np.nan)),
            ("TS (MPa, Input/Predicted)", user_p.get('TS_mpa', np.nan)),
            ("HB (Input/Predicted)", user_p.get('HB', np.nan)),
            ("HB (Model Input)", en_p.get('HB_processed_for_prediction', np.nan)),
            ("ν (Input)", user_p.get('nu', np.nan)),
            ("AI: σf' (MPa)", en_p.get('sigma_f_prime', np.nan)),
            ("AI: b", en_p.get('b_fatigue', np.nan)),
            ("AI: εf'", en_p.get('epsilon_f_prime', np.nan)),
            ("AI: c", en_p.get('c_fatigue', np.nan)),
        ]
        if phys_p:
            all_params_list.extend([
                (f"Physics ({phys_p.get('method_name', 'N/A')}): σf' (MPa)", phys_p.get('spf_MPa_phys', np.nan)),
                (f"Physics ({phys_p.get('method_name', 'N/A')}): εf'", phys_p.get('epf_phys', np.nan)),
            ])
        if shear_p:
            all_params_list.extend([
                (f"Shear ({shear_p.get('conversion_method', 'N/A')}): τf' (MPa)", shear_p.get('tauf_MPa', np.nan)),
                (f"Shear ({shear_p.get('conversion_method', 'N/A')}): b0", shear_p.get('b0', np.nan)),
                (f"Shear ({shear_p.get('conversion_method', 'N/A')}): γf'", shear_p.get('gammaf', np.nan)),
                (f"Shear ({shear_p.get('conversion_method', 'N/A')}): c0", shear_p.get('c0', np.nan)),
                (f"Shear ({shear_p.get('conversion_method', 'N/A')}): G (MPa)", shear_p.get('G_mpa', np.nan)),
            ])
        
        df_all_params = pd.DataFrame(all_params_list, columns=['Parameter', 'Value'])
        # Format float values
        for col in df_all_params.columns:
            if df_all_params[col].dtype == 'object': # Check if column might contain floats
                try:
                    df_all_params[col] = pd.to_numeric(df_all_params[col], errors='ignore')
                except: pass # Ignore if conversion fails for mixed types like 'Parameter'
        
        float_cols = df_all_params.select_dtypes(include=np.number).columns
        format_dict = {col: lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else x for col in float_cols}

        st.download_button(
            label="모든 예측/계산 파라미터 CSV 다운로드",
            data=df_all_params.to_csv(index=False, float_format='%.4f').encode('utf-8-sig'), # utf-8-sig for Excel
            file_name="predicted_fatigue_parameters.csv",
            mime='text/csv',
            use_container_width=True
        )

    with tabs[1]: # Tensile Curve Tab
        st.subheader("ϵ-N (인장 변형률-수명) 곡선")
        en_data = st.session_state.en_results
        fig_en, ax_en = plt.subplots(figsize=(10, 6))
        
        Nf_plot = en_data.get("Nf", np.array([])) * 2 # Plot against 2Nf (Reversals)
        total_strain = en_data.get("total_strain_amplitude")
        elastic_strain = en_data.get("elastic_strain_amplitude")
        plastic_strain = en_data.get("plastic_strain_amplitude")

        plot_valid = False
        if total_strain is not None and not np.all(np.isnan(total_strain)):
            ax_en.loglog(Nf_plot, total_strain, '-', label='Total Strain (ε_a,pred)', linewidth=2)
            if elastic_strain is not None: ax_en.loglog(Nf_plot, elastic_strain, '--', label='Elastic Strain (pred)', alpha=0.8)
            if plastic_strain is not None: ax_en.loglog(Nf_plot, plastic_strain, ':', label='Plastic Strain (pred)', alpha=0.8)
            plot_valid = True
            
        if plot_valid:
            ax_en.set_xlabel('Reversals to Failure (2Nf)')
            ax_en.set_ylabel('Strain Amplitude (ε_a)')
            ax_en.legend()
            ax_en.grid(True, which="both", ls="-", alpha=0.7)
            valid_strains = total_strain[~np.isnan(total_strain)] # Use original total_strain for limits
            if len(valid_strains) > 0:
                ax_en.set_ylim(bottom=max(1e-5, np.min(valid_strains[valid_strains > 0]) * 0.5), top=np.max(valid_strains) * 1.5)
            else:
                ax_en.set_ylim(bottom=1e-5)
            ax_en.set_title('Predicted E-N (Strain-Life) Curve')
        else:
            ax_en.text(0.5, 0.5, 'E-N Curve data not available or contains NaNs.', ha='center', va='center', transform=ax_en.transAxes)
            ax_en.set_title('Predicted E-N (Strain-Life) Curve (Data N/A)')
        
        st.pyplot(fig_en)

        df_en_curve = pd.DataFrame({
            'Reversals_2Nf': Nf_plot,
            'Total_Strain_Amplitude': total_strain,
            'Elastic_Strain_Amplitude': elastic_strain,
            'Plastic_Strain_Amplitude': plastic_strain,
            'Stress_Amplitude_MPa': en_data.get("stress_amplitude")
        })
        csv_en_curve = df_en_curve.to_csv(index=False, float_format='%.5e').encode('utf-8-sig')
        
        img_en_buf = io.BytesIO()
        fig_en.savefig(img_en_buf, format="png", dpi=300)
        img_en_buf.seek(0)

        dl_col1_en, dl_col2_en = st.columns(2)
        with dl_col1_en:
            st.download_button(label="E-N 곡선 데이터 (CSV) 다운로드", data=csv_en_curve, file_name="en_curve_data.csv", mime="text/csv", use_container_width=True, disabled=not plot_valid)
        with dl_col2_en:
            st.download_button(label="E-N 곡선 이미지 (PNG) 다운로드", data=img_en_buf, file_name="en_curve_plot.png", mime="image/png", use_container_width=True, disabled=not plot_valid)

        with st.expander("E-N 곡선 수치 데이터 보기"):
            if plot_valid and not df_en_curve.empty:
                st.dataframe(df_en_curve.style.format("{:.4e}"))
            else:
                st.write("표시할 E-N 곡선 데이터가 없습니다.")

    with tabs[2]: # Shear Curve Tab
        st.subheader("γ-N (전단 변형률-수명) 곡선")
        shear_results_gn = st.session_state.shear_results
        en_base_data_gn = st.session_state.en_results # For Nf
        
        fig_gn, ax_gn = plt.subplots(figsize=(10, 6))
        plot_valid_gn = False
        df_gn_curve_plot = pd.DataFrame() # Initialize

        if shear_results_gn and en_base_data_gn:
            Nf_gn_plot = en_base_data_gn.get("Nf", np.array([])) * 2 # Reversals (2Nf)

            tauf_prime = shear_results_gn.get('tauf_MPa')
            gammaf_prime = shear_results_gn.get('gammaf')
            b0 = shear_results_gn.get('b0')
            c0 = shear_results_gn.get('c0')
            G_mpa = shear_results_gn.get('G_mpa')

            if all(val is not None and not np.isnan(val) for val in [tauf_prime, gammaf_prime, b0, c0, G_mpa]) and G_mpa > 0 and len(Nf_gn_plot) > 0:
                el_shear_strain = (tauf_prime / G_mpa) * (Nf_gn_plot / 2)**b0 # Nf_gn_plot is 2Nf, formula uses Nf or 2Nf depending on b0 def. Assuming b0 is for 2Nf.
                pl_shear_strain = gammaf_prime * (Nf_gn_plot / 2)**c0         # Let's assume b0, c0 are for Nf (cycles)
                                                                              # Original code used reversals_gn_plot ** b0_gn which was Nf.
                                                                              # The Coffin-Manson is usually (2Nf)^b or (Nf)^b.
                                                                              # The plot X-axis is 2Nf.
                                                                              # If b,c are for 2Nf: (tau_f'/G)*(2Nf)^b + gamma_f'*(2Nf)^c
                                                                              # If b,c are for Nf: (tau_f'/G)*(Nf)^b + gamma_f'*(Nf)^c
                                                                              # The E-N curve used (2*Nf)**b. So shear should be consistent.
                
                # Assuming b0, c0 are exponents for 2Nf (reversals)
                elastic_shear_strain_gn = (tauf_prime / G_mpa) * (Nf_gn_plot)**b0
                plastic_shear_strain_gn = gammaf_prime * (Nf_gn_plot)**c0
                total_shear_strain_gn = elastic_shear_strain_gn + plastic_shear_strain_gn
                
                ax_gn.loglog(Nf_gn_plot, total_shear_strain_gn, '-', label='Total Shear Strain (γ_a,pred)', linewidth=2)
                ax_gn.loglog(Nf_gn_plot, elastic_shear_strain_gn, '--', label='Elastic Shear Strain (pred)', alpha=0.8)
                ax_gn.loglog(Nf_gn_plot, plastic_shear_strain_gn, ':', label='Plastic Shear Strain (pred)', alpha=0.8)
                plot_valid_gn = True

                df_gn_curve_plot = pd.DataFrame({
                    'Reversals_2Nf': Nf_gn_plot,
                    'Total_Shear_Strain_Amplitude': total_shear_strain_gn,
                    'Elastic_Shear_Strain_Amplitude': elastic_shear_strain_gn,
                    'Plastic_Shear_Strain_Amplitude': plastic_shear_strain_gn
                })
        
        if plot_valid_gn:
            ax_gn.set_xlabel('Reversals to Failure (2Nf)')
            ax_gn.set_ylabel('Shear Strain Amplitude (γ_a)')
            ax_gn.legend()
            ax_gn.grid(True, which="both", ls="-", alpha=0.7)
            valid_strains_gn = df_gn_curve_plot['Total_Shear_Strain_Amplitude'].dropna()
            if not valid_strains_gn.empty:
                 ax_gn.set_ylim(bottom=max(1e-5, valid_strains_gn[valid_strains_gn > 0].min() * 0.5), top=valid_strains_gn.max() * 1.5)
            else:
                ax_gn.set_ylim(bottom=1e-5)
            ax_gn.set_title('Predicted γ-N (Shear Strain-Life) Curve')
            st.caption(f"전단 변환 방법: {shear_results_gn.get('conversion_method', 'N/A')}")
        else:
            ax_gn.text(0.5, 0.5, 'γ-N Curve data not available or parameters invalid.', ha='center', va='center', transform=ax_gn.transAxes)
            ax_gn.set_title('Predicted γ-N (Shear Strain-Life) Curve (Data N/A)')
            if not shear_results_gn or np.isnan(shear_results_gn.get('G_mpa', np.nan)) or shear_results_gn.get('G_mpa', 0) <=0 :
                 st.warning("전단 피로 파라미터 계산에 실패했거나 전단 탄성 계수(G)가 유효하지 않아 γ-N 곡선을 생성할 수 없습니다.")
        
        st.pyplot(fig_gn)

        csv_gn_curve = df_gn_curve_plot.to_csv(index=False, float_format='%.5e').encode('utf-8-sig')
        img_gn_buf = io.BytesIO()
        fig_gn.savefig(img_gn_buf, format="png", dpi=300)
        img_gn_buf.seek(0)

        dl_gn_col1, dl_gn_col2 = st.columns(2)
        with dl_gn_col1:
            st.download_button(label="γ-N 곡선 데이터 (CSV) 다운로드", data=csv_gn_curve, file_name="gamma_n_curve_data.csv", mime="text/csv", use_container_width=True, disabled=not plot_valid_gn)
        with dl_gn_col2:
            st.download_button(label="γ-N 곡선 이미지 (PNG) 다운로드", data=img_gn_buf, file_name="gamma_n_curve_plot.png", mime="image/png", use_container_width=True, disabled=not plot_valid_gn)

        with st.expander("γ-N 곡선 수치 데이터 보기"):
            if plot_valid_gn and not df_gn_curve_plot.empty:
                 st.dataframe(df_gn_curve_plot.style.format("{:.4e}"))
            else:
                st.write("표시할 γ-N 곡선 데이터가 없습니다.")
else:
    st.info('👈 사이드바에서 입력 모드를 선택하고, 필요한 값을 입력한 후 "피로 거동 예측 실행" 버튼을 누르세요.')

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 0.8em;">
        © 2024 YeoJoon Yoon. All Rights Reserved.<br>
        Contact: <a href="mailto:goat@sogang.ac.kr">goat@sogang.ac.kr</a>
    </div>
    """,
    unsafe_allow_html=True
)

# streamlit run FatiguePredictor0529.py