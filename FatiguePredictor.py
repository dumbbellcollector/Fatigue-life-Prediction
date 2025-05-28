import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import os
import pandas as pd
import io # For image download

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
def all_inputs_valid(e_mod, ys, ts, hb_input, poisson_ratio):
    if e_mod is None or e_mod <= 0: return False
    if ys is None or ys <= 0: return False
    if ts is None or ts <= 0: return False
    # HB can be 0 or None if not used for physics, but for prediction it's better to have a value.
    # The predict_fatigue_curves_hybrid handles HB_val being None/nan, so direct check here might be optional
    # depending on strictness. For now, let's assume it should be non-negative if provided.
    if hb_input is not None and hb_input < 0: return False # Allow 0, but not negative
    if poisson_ratio is None or not (0 <= poisson_ratio <= 0.5): return False
    return True

# --- main.ipynb의 inverse_transform_targets 함수 ---
def inverse_transform_targets(y_scaled_data, scalers_y_dict, target_cols_list):
    if y_scaled_data.ndim == 1:
        y_scaled_data = y_scaled_data.reshape(1, -1)
        
    y_transformed_individually = np.zeros_like(y_scaled_data)
    for i, col_name in enumerate(target_cols_list):
        if col_name not in scalers_y_dict: # 키 존재 여부 확인
            raise KeyError(f"Scaler for target '{col_name}' not found in scalers_y_dict. Available keys: {list(scalers_y_dict.keys())}")
        y_transformed_individually[:, i] = scalers_y_dict[col_name].inverse_transform(y_scaled_data[:, i].reshape(-1, 1)).flatten()
    
    y_orig_scale = y_transformed_individually.copy()
    
    # Cell 6의 기준에 따라, 로그 변환된 컬럼 이름은 'epf'로 가정합니다.
    log_col_expected_name = 'epf' 
    fallback_log_col_name = 'epf_log' # 혹시 'epf_log'로 저장된 pkl을 위한 대체 경로

    if log_col_expected_name in target_cols_list:
        try:
            current_log_col_idx = target_cols_list.index(log_col_expected_name)
            y_orig_scale[:, current_log_col_idx] = np.expm1(y_transformed_individually[:, current_log_col_idx])
        except ValueError:
            st.warning(f"'{log_col_expected_name}' was in target_cols_list but index could not be found. Skipping expm1 transformation for it.")
    elif fallback_log_col_name in target_cols_list: # 'epf'가 없고 'epf_log'가 있는 경우
        try:
            current_fallback_log_col_idx = target_cols_list.index(fallback_log_col_name)
            st.info(f"Note: Found '{fallback_log_col_name}' in target_cols and applying expm1. "
                    f"The primary expected log-column name (based on notebook Cell 6 assumption) is '{log_col_expected_name}'.")
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

# --- Helper function for Shear Conversion (New) ---
def convert_to_shear_parameters(spf_prime, b_fatigue, epf_prime, c_fatigue, E_gpa, TS_mpa, nu=0.3):
    tau_vm = spf_prime / np.sqrt(3)
    gamma_vm = np.sqrt(3) * epf_prime
    
    tau_mp = spf_prime / (1 + nu) 
    gamma_mp = 2 * epf_prime

    b0 = b_fatigue 
    c0 = c_fatigue

    conversion_method = "Unknown"
    tauf_prime, gammaf_prime = np.nan, np.nan

    if TS_mpa <= 1100:
        tauf_prime, gammaf_prime = tau_vm, gamma_vm
        conversion_method = "von Mises Criteria"
    elif TS_mpa >= 1696:
        tauf_prime, gammaf_prime = tau_mp, gamma_mp
        conversion_method = "Maximum Principal Stress/Strain Criteria"
    else:
        alpha = (TS_mpa - 1100) / (1696 - 1100)
        tauf_prime = (1 - alpha) * tau_vm + alpha * tau_mp
        gammaf_prime = (1 - alpha) * gamma_vm + alpha * gamma_mp
        conversion_method = f"Interpolated (von Mises to Max Principal, α={alpha:.2f})"

    shear_params = {
        'tauf_MPa': tauf_prime,
        'gammaf': gammaf_prime,
        'b0': b0,
        'c0': c0,
        'conversion_method': conversion_method
    }
    return shear_params

# --- Hybrid Model Prediction and Curve Generation (Modified) ---
def predict_fatigue_curves_hybrid(E_val_gpa, YS_val_mpa, TS_val_mpa, HB_val, model, scaler_X, 
                                  scalers_y_dict, target_cols_list,
                                  device, nu=0.3): 
    
    E_val_mpa = E_val_gpa * 1000 

    hb_processed_val = HB_val
    if HB_val is None or np.isnan(HB_val):
        if TS_val_mpa is not None and not np.isnan(TS_val_mpa):
            hb_processed_val = 1.8 * TS_val_mpa + 105 
        else: 
             raise ValueError("HB 값과 TS_MPa 값 모두 제공되지 않아 HB_processed를 계산할 수 없습니다.")

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

    # Cell 6 기준: 'epf'가 로그 변환된 컬럼의 이름으로 params_dict에 키로 존재
    # inverse_transform_targets 함수가 이미 해당 컬럼에 대해 np.expm1을 적용했음.
    if 'epf' in params_dict: # 노트북 Cell 6의 pkl 저장 기준
        epsilon_f_prime = params_dict.get('epf')
    elif 'epf_log' in params_dict: # 'epf_log'로 저장된 pkl을 위한 대체 경로
        epsilon_f_prime = params_dict.get('epf_log')
    else:
        epsilon_f_prime = None 

    c_fatigue = params_dict.get('c')

    if sigma_f_prime is None or b_fatigue is None or epsilon_f_prime is None or c_fatigue is None:
        # 오류 메시지에서 'epf_log/epf' 대신, 실제로 찾으려고 시도한 키를 명시하는 것이 더 정확할 수 있음
        # 예를 들어, target_cols_list에 있는 epf 관련 키 ('epf' 또는 'epf_log')를 사용
        epf_key_in_list = 'epf' if 'epf' in target_cols_list else 'epf_log' if 'epf_log' in target_cols_list else 'epf/epf_log'
        missing_keys = [key for key, val in zip(['spf_MPa', 'b', epf_key_in_list, 'c'], 
                                               [sigma_f_prime, b_fatigue, epsilon_f_prime, c_fatigue]) if val is None]
        raise ValueError(f"모델 예측에서 다음 필수 파라미터들을 얻을 수 없습니다: {missing_keys}. "
                         f"사용된 target_cols_list: {target_cols_list}, params_dict에 있는 키: {list(params_dict.keys())}")

    # E-N Curve (Tensile)
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
        "E_mpa": E_val_mpa
    }
    return results

# --- Load Model and Scalers ---
@st.cache_resource
def load_resources(model_path=MODEL_PATH, # 전역 상수 사용
                   scaler_x_path=SCALER_X_PATH, # 전역 상수 사용
                   scaler_y_path=SCALER_Y_PATH): # 전역 상수 사용
    # 모델 로드
    model_hidden_dims = [192, 384, 352, 224]
    model_dropout_p = 0.35
    input_dim = 4 
    output_dim = 4 

    model = FatiguePINN(input_dim=input_dim, 
                        output_dim=output_dim, 
                        hidden_dims=model_hidden_dims, 
                        dropout_p=model_dropout_p)
    
    # 전역 device 변수 사용
    _device = device 

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Scaler X file not found: {scaler_x_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler Y file not found: {scaler_y_path}")

    if torch.cuda.is_available() and _device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
    elif torch.backends.mps.is_available() and _device.type == 'mps':
        model.load_state_dict(torch.load(model_path, map_location='mps'))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(_device)
    model.eval()

    scaler_X = joblib.load(scaler_x_path)
    
    data_y = joblib.load(scaler_y_path)
    
    # Cell 6 기준: pkl 파일은 'epf'를 로그 변환된 컬럼명으로 저장한다고 가정
    expected_cols_for_model = ['spf_MPa', 'b', 'epf', 'c'] 

    if isinstance(data_y, dict) and 'scalers' in data_y and 'target_cols' in data_y:
        scalers_y_dict = data_y['scalers']
        target_cols_list = data_y['target_cols']
        
        if set(target_cols_list) != set(expected_cols_for_model):
            # 순서는 다를 수 있으나, 포함된 이름 자체가 다른 경우 경고
            # 특히 'epf' 대신 'epf_log'가 있거나 그 반대인 경우를 감지하여 안내
            if 'epf' in target_cols_list and 'epf_log' not in target_cols_list and 'epf_log' in expected_cols_for_model:
                 # 이 경우는 expected_cols_for_model이 'epf_log'를 기대했으나 실제로는 'epf'만 있을 때 (현재 로직상 발생 안 함)
                 pass 
            elif 'epf_log' in target_cols_list and 'epf' not in target_cols_list and 'epf' in expected_cols_for_model:
                # 실제 pkl에 'epf_log'가 있고, 우리는 'epf'를 기대하는 경우 (Cell 6 기준과 반대)
                st.warning(
                    f"경고: 로드된 target_cols ({target_cols_list})에 'epf_log'가 포함되어 있습니다. "
                    f"현재 앱은 노트북 Cell 6 기준에 따라 '{expected_cols_for_model}' (즉, 'epf')를 기대합니다. "
                    f"inverse_transform_targets 함수에서 'epf_log'를 대체 처리하려고 시도하지만, "
                    f"정확한 작동을 위해 노트북에서 target_cols에 'epf'를 사용하고 scalers_y.pkl을 재생성하는 것을 권장합니다."
                )
            else: # 그 외 일반적인 이름 불일치
                 st.warning(
                    f"경고: 로드된 target_cols ({target_cols_list})가 예상 기본값 ({expected_cols_for_model})과 다릅니다. "
                    f"로드된 값을 사용하나, 예상치 못한 동작이 발생할 수 있습니다. 노트북의 scalers_y.pkl 저장 부분을 확인하세요."
                )
        # 순서 일치 여부도 중요할 수 있으므로, 리스트 직접 비교도 고려 (선택적)
        if target_cols_list != expected_cols_for_model and set(target_cols_list) == set(expected_cols_for_model):
            st.info(f"로드된 target_cols ({target_cols_list})의 순서가 예상된 순서 ({expected_cols_for_model})와 다릅니다. 이름은 일치하므로 계속 진행합니다.")

        missing_scalers = [col for col in target_cols_list if col not in scalers_y_dict]
        if missing_scalers:
            raise ValueError(f"'{scaler_y_path}' 파일의 'scalers' 딕셔너리에 다음 타겟 컬럼에 대한 스케일러가 없습니다: {missing_scalers}. "
                             f"'target_cols'는 {target_cols_list} 입니다.")

    elif isinstance(data_y, dict):
        st.warning(f"경고: '{scaler_y_path}' 파일이 이전 형식으로 보입니다 (스케일러 딕셔너리만 포함). "
                   f"'target_cols' 정보가 없어 모델 학습 시 예상되는 기본값 {expected_cols_for_model}을 사용합니다. "
                   f"정확한 작동을 위해 노트북에서 'scalers_y.pkl' 저장 방식을 {'{'}'scalers': ..., 'target_cols': ...{'}'} 형태로 업데이트하고, "
                   f"target_cols를 '{expected_cols_for_model}'로 저장하는 것을 강력히 권장합니다.")
        scalers_y_dict = data_y 
        target_cols_list = expected_cols_for_model 
        missing_keys_in_scaler = [key for key in target_cols_list if key not in scalers_y_dict]
        if missing_keys_in_scaler:
            raise ValueError(f"이전 형식의 '{scaler_y_path}' 파일에서 로드된 스케일러 딕셔너리에 "
                             f"예상되는 기본 타겟 컬럼({expected_cols_for_model})에 대한 키가 없습니다. "
                             f"누락된 키: {missing_keys_in_scaler}")
    else:
        raise ValueError(f"'{scaler_y_path}' 파일의 형식을 인식할 수 없습니다. "
                         f"{'{'}'scalers': ..., 'target_cols': ...{'}'} 형식 또는 스케일러 딕셔너리 형식을 기대합니다.")

    if model.output_dim != len(target_cols_list):
        st.error(f"모델의 출력 차원({model.output_dim})과 로드/결정된 타겟 컬럼의 수({len(target_cols_list)})가 일치하지 않습니다. "
                 f"모델 아키텍처 또는 'scalers_y.pkl' 파일의 'target_cols'를 확인하세요. 사용된 타겟 컬럼: {target_cols_list}")
        raise ValueError("Model output dimension and target columns mismatch.")
        
    return model, scaler_X, scalers_y_dict, target_cols_list

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Fatigue Life Predictor")
st.title('Fatigue Life Predictor (ε-N / γ-N)')
st.write("Enter Monotonic Properties and Select mode, Get Prediction.")

# --- Session State 초기화 ---
if 'prediction_triggered' not in st.session_state:
    st.session_state.prediction_triggered = False
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'en_results' not in st.session_state:
    st.session_state.en_results = None
if 'shear_results' not in st.session_state:
    st.session_state.shear_results = None
if 'physics_params' not in st.session_state:
    st.session_state.physics_params = None
if 'current_prediction_mode' not in st.session_state:
    st.session_state.current_prediction_mode = 'tensile'

# 리소스 로드 (앱 시작 시 한 번)
model, scaler_X, scalers_y_dict, target_cols_list = load_resources()

if model is None: st.stop()

# 사이드바에 입력 섹션 배치
with st.sidebar:
    st.header("입력 파라미터")
    st.subheader("재료 특성")
    e_mod_gpa_input = st.number_input('탄성 계수 (E, GPa)', min_value=1.0, value=200.0, format='%.1f', help="GPa 단위로 입력하세요. (예: 강철 약 200 GPa)")
    ys_mpa_input = st.number_input('항복 강도 (YS, MPa)', min_value=1.0, value=500.0, format='%.1f')
    ts_mpa_input = st.number_input('인장 강도 (UTS, MPa)', min_value=1.0, value=700.0, format='%.1f')
    hb_input_val = st.number_input('브리넬 경도 (HB)', min_value=0.0, value=200.0, format='%.1f', help="물리 기반 spf/epf 계산에 필요합니다.")
    poisson_ratio_input = st.number_input("포아송 비 (ν)", min_value=0.0, max_value=0.5, value=0.3, step=0.01, format='%.2f', help="전단 계산에 사용됩니다")

    if hb_input_val <= 0:
         st.warning("유효한 HB 값을 입력하세요. HB 값이 0 이하이면 물리 기반 경험식 계산 및 일부 예측 정확도에 영향을 줄 수 있습니다.")

    if st.button("피로 거동 예측 실행", use_container_width=True, type="primary"):
        st.session_state.prediction_triggered = True
        
        if not all_inputs_valid(e_mod_gpa_input * 1000, ys_mpa_input, ts_mpa_input, hb_input_val, poisson_ratio_input):
            st.error("모든 필수 입력값을 올바르게 입력해주세요. (E, YS, UTS > 0, 0 <= ν <= 0.5)")
            st.session_state.en_results = None 
            st.session_state.shear_results = None
            st.session_state.physics_params = None
        else:
            try:
                e_mod_mpa_for_calc = e_mod_gpa_input * 1000 # 내부 계산용 MPa 값
                temp_user_inputs = {
                    'E_gpa': e_mod_gpa_input, # GPa로 저장 (사용자 입력값)
                    'E_mpa': e_mod_mpa_for_calc, # MPa로 저장 (계산용)
                    'YS_mpa': ys_mpa_input, 'TS_mpa': ts_mpa_input,
                    'HB': hb_input_val, 'nu': poisson_ratio_input
                }
                current_target_cols = target_cols_list 
                hb_to_pass = hb_input_val if hb_input_val is not None and hb_input_val > 0 else np.nan

                predicted_en_results = predict_fatigue_curves_hybrid(
                    e_mod_gpa_input, ys_mpa_input, ts_mpa_input, hb_to_pass, 
                    model, scaler_X, scalers_y_dict, 
                    current_target_cols, device
                )
                
                spf_phys, epf_phys, phys_method_name = get_physics_params(
                    hb_input_val if hb_input_val > 0 else np.nan, 
                    ts_mpa_input,
                    e_mod_mpa_for_calc # get_physics_params는 MPa 단위 E를 받음
                )
                temp_physics_params = {
                    'spf_MPa_phys': spf_phys, 'epf_phys': epf_phys, 'method_name': phys_method_name
                }

                shear_calc_results = convert_to_shear_parameters(
                    spf_prime=predicted_en_results['sigma_f_prime'],
                    b_fatigue=predicted_en_results['b_fatigue'],
                    epf_prime=predicted_en_results['epsilon_f_prime'],
                    c_fatigue=predicted_en_results['c_fatigue'],
                    E_gpa=e_mod_gpa_input, TS_mpa=ts_mpa_input, nu=poisson_ratio_input 
                )
                G_mpa_calc = (predicted_en_results['E_mpa']) / (2 * (1 + poisson_ratio_input))
                shear_calc_results['G_mpa'] = G_mpa_calc
                
                st.session_state.user_inputs = temp_user_inputs
                st.session_state.en_results = predicted_en_results
                st.session_state.physics_params = temp_physics_params
                st.session_state.shear_results = shear_calc_results

                #st.sidebar.success("예측 및 계산 완료!")

            except FileNotFoundError as fe:
                st.sidebar.error(f"필수 파일을 찾을 수 없습니다: {fe}.")
                st.session_state.en_results = None
                st.session_state.shear_results = None
                st.session_state.physics_params = None
            except ValueError as ve:
                st.sidebar.error(f"입력값 또는 데이터 처리 중 오류: {ve}")
                st.session_state.en_results = None
                st.session_state.shear_results = None
                st.session_state.physics_params = None
            except Exception as e:
                st.sidebar.error(f"예측 중 예기치 않은 오류: {e}")
                st.session_state.en_results = None
                st.session_state.shear_results = None
                st.session_state.physics_params = None

# --- Main Area for Results ---
if st.session_state.prediction_triggered and st.session_state.en_results:
    # Create tabs: "모델 예측 파라미터", "인장 곡선 (E-N)", "전단 곡선 (γ-N)"
    # "입력 요약" 탭 제거
    tab_titles = ["피로 파라미터", "인장 곡선 (E-N)", "전단 곡선 (γ-N)"]
    # if st.session_state.current_prediction_mode == 'shear' and st.session_state.shear_results: # 이 조건은 이제 항상 참이거나, shear_results 유무로 판단
    # 항상 전단 탭을 만들도록 하되, 내용 표시는 shear_results 유무로 결정
    
    tabs = st.tabs(tab_titles)

    # Tab 1: Predicted Parameters (기존 Tab 2의 내용 + 수식)
    with tabs[0]:
        st.subheader("인장 피로 파라미터 예측 결과")
        en_params = st.session_state.en_results
        phys_params = st.session_state.physics_params
        
        col1, col2 = st.columns(2)
        with col1:
            # method_name_display = phys_params.get('method_name', '경험식') # 삭제
            # st.markdown(f"{method_name_display} 기반 σf'") # 삭제
            st.metric(r"물리 식 계산 피로 강도 계수 ($\sigma_f'$, Calculated Fatigue Strength Coeff.)", f"{phys_params.get('spf_MPa_phys', np.nan):.2f} MPa", 
                      help=f"입력된 HB값과 UTS를 사용하여 경험식으로 계산됩니다. (방법: {phys_params.get('method_name', 'N/A')})")
            st.metric(r"피로 강도 계수 ($\sigma_f'$, Fatigue Strength Coefficient)", f"{en_params.get('sigma_f_prime', np.nan):.2f} MPa")
            st.metric("피로 강도 지수 (b, Fatigue Strength Exponent)", f"{en_params.get('b_fatigue', np.nan):.4f}")

        with col2:
            # method_name_display = phys_params.get('method_name', '경험식') # 삭제
            # st.markdown(f"{method_name_display} 기반 εf'") # 삭제
            st.metric(r"물리 식 계산 피로 연성 계수 ($\epsilon_f'$, Calculated Fatigue Ductility Coeff.)", f"{phys_params.get('epf_phys', np.nan):.4f}",
                      help=f"입력된 HB값과 UTS, E를 사용하여 경험식으로 계산됩니다. (방법: {phys_params.get('method_name', 'N/A')})")
            st.metric(r"피로 연성 계수 ($\epsilon_f'$, Fatigue Ductility Coefficient)", f"{en_params.get('epsilon_f_prime', np.nan):.4f}")
            st.metric("피로 연성 지수 (c, Fatigue Ductility Exponent)", f"{en_params.get('c_fatigue', np.nan):.4f}")
        
        st.markdown("---_인장 Coffin-Manson 관계식_---")
        st.latex(r"\frac{\Delta\epsilon}{2} = \frac{\sigma'_f}{E}\,(2N_f)^b + \epsilon'_f\,(2N_f)^c")

        all_params_data = {
            'Parameter': [
                r"AI: 피로 강도 계수 ($\sigma_f'$, Fatigue Strength Coefficient, MPa)", 
                "AI: 피로 강도 지수 (b, Fatigue Strength Exponent)",
                r"AI: 피로 연성 계수 ($\epsilon_f'$, Fatigue Ductility Coefficient)", 
                "AI: 피로 연성 지수 (c, Fatigue Ductility Exponent)",
                r"Physics: 추정된 피로 강도 계수 ($\sigma_f'$, Estimated Fatigue Strength Coeff., MPa)", 
                r"Physics: 추정된 피로 연성 계수 ($\epsilon_f'$, Estimated Fatigue Ductility Coeff.)",
                "Physics: 사용된 경험식 방법 (Method)"
            ],
            'Value': [
                f"{en_params.get('sigma_f_prime', np.nan):.2f}", 
                f"{en_params.get('b_fatigue', np.nan):.4f}",
                f"{en_params.get('epsilon_f_prime', np.nan):.4f}", 
                f"{en_params.get('c_fatigue', np.nan):.4f}",
                f"{phys_params.get('spf_MPa_phys', np.nan):.2f}", 
                f"{phys_params.get('epf_phys', np.nan):.4f}",
                str(phys_params.get('method_name', 'N/A'))
            ]
        }

        # 전단 파라미터 및 수식 추가 (st.session_state.shear_results 유무에 따라)
        if st.session_state.shear_results:
            st.divider()
            st.subheader("전단 피로 파라미터 변환")
            shear_params = st.session_state.shear_results
            s_col1, s_col2 = st.columns(2)
            s_col1.metric(r"전단 피로 강도 계수 ($\tau_f'$, Shear Fatigue Strength Coeff.)", f"{shear_params.get('tauf_MPa', np.nan):.2f} MPa")
            s_col1.metric(r"전단 피로 강도 지수 ($b_0$, Shear Fatigue Strength Exp.)", f"{shear_params.get('b0', np.nan):.4f}")
            s_col2.metric(r"전단 피로 연성 계수 ($\gamma_f'$, Shear Fatigue Ductility Coeff.)", f"{shear_params.get('gammaf', np.nan):.4f}")
            s_col2.metric(r"전단 피로 연성 지수 ($c_0$, Shear Fatigue Ductility Exp.)", f"{shear_params.get('c0', np.nan):.4f}")
            st.caption(f"전단 변환 방법 (Conversion Method): {shear_params.get('conversion_method', 'N/A')}")
            st.caption(f"계산된 전단 탄성 계수 (G, Shear Modulus): {shear_params.get('G_mpa', np.nan):.0f} MPa")
            
            st.markdown("---_전단 Coffin-Manson 관계식 (유사 형태)_---")
            st.latex(r"\frac{\Delta\gamma}{2} = \frac{\tau'_f}{G}\,(2N_f)^{b_0} + \gamma'_f\,(2N_f)^{c_0}")

            all_params_data['Parameter'].extend([
                r"Shear: 전단 피로 강도 계수 ($\tau_f'$, Shear Fatigue Strength Coeff., MPa)", 
                r"Shear: 전단 피로 강도 지수 ($b_0$, Shear Fatigue Strength Exp.)",
                r"Shear: 전단 피로 연성 계수 ($\gamma_f'$, Shear Fatigue Ductility Coeff.)", 
                r"Shear: 전단 피로 연성 지수 ($c_0$, Shear Fatigue Ductility Exp.)",
                "Shear: 전단 변환 방법 (Conversion Method)", 
                "Shear: 전단 탄성 계수 (G, Shear Modulus, MPa)"
            ])
            all_params_data['Value'].extend([
                f"{shear_params.get('tauf_MPa', np.nan):.2f}", 
                f"{shear_params.get('b0', np.nan):.4f}",
                f"{shear_params.get('gammaf', np.nan):.4f}", 
                f"{shear_params.get('c0', np.nan):.4f}",
                str(shear_params.get('conversion_method', 'N/A')), 
                f"{shear_params.get('G_mpa', np.nan):.0f}"
            ])
        
        df_all_params = pd.DataFrame(all_params_data)
        csv_all_params = df_all_params.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="모든 예측/계산 파라미터 CSV 다운로드",
            data=csv_all_params,
            file_name="predicted_fatigue_parameters.csv",
            mime='text/csv',
            use_container_width=True # 버튼 폭 채우기
        )

    # Tab 2: Tensile Curve (E-N) (기존 Tab 3)
    with tabs[1]:
        st.subheader("E-N (인장 변형률-수명) 곡선")
        en_data = st.session_state.en_results
        fig_en, ax_en = plt.subplots(figsize=(10, 6))
        
        reversals_en = en_data.get("Nf", np.array([]))
        total_strain_en = en_data.get("total_strain_amplitude")
        elastic_strain_en = en_data.get("elastic_strain_amplitude")
        plastic_strain_en = en_data.get("plastic_strain_amplitude")

        if total_strain_en is not None and not np.all(np.isnan(total_strain_en)):
            ax_en.loglog(reversals_en, total_strain_en, '-', label='Total Strain (ε_a,pred)', linewidth=2)
            if elastic_strain_en is not None: ax_en.loglog(reversals_en, elastic_strain_en, '--', label='Elastic Strain (pred)', alpha=0.8)
            if plastic_strain_en is not None: ax_en.loglog(reversals_en, plastic_strain_en, ':', label='Plastic Strain (pred)', alpha=0.8)
            
            ax_en.set_xlabel('Reversals to Failure (2Nf)')
            ax_en.set_ylabel('Strain Amplitude (ε_a)')
            ax_en.legend()
            ax_en.grid(True, which="both", ls="-", alpha=0.7)
            valid_strains = total_strain_en[~np.isnan(total_strain_en)]
            if len(valid_strains) > 0:
                ax_en.set_ylim(bottom=max(1e-5, np.min(valid_strains) * 0.5), top=np.max(valid_strains) * 1.5)
            else:
                ax_en.set_ylim(bottom=1e-5)
            ax_en.set_title('Predicted E-N (Strain-Life) Curve')
        else:
            ax_en.text(0.5, 0.5, 'E-N Curve data not available or contains NaNs.', ha='center', va='center', transform=ax_en.transAxes)
            ax_en.set_title('Predicted E-N (Strain-Life) Curve (Data N/A)')
        
        # st.latex(r"\\frac{\\Delta\\epsilon}{2} = \\frac{\\sigma'_f}{E}\\,(2N_f)^b + \\epsilon'_f\\,(2N_f)^c") # 파라미터 탭으로 이동
        st.pyplot(fig_en)

        # Download buttons for E-N curve
        df_en_curve = pd.DataFrame({
            '2Nf': reversals_en,
            'Total_Strain_Amplitude': total_strain_en,
            'Elastic_Strain_Amplitude': elastic_strain_en,
            'Plastic_Strain_Amplitude': plastic_strain_en,
            'Stress_Amplitude_MPa': en_data.get("stress_amplitude")
        })
        csv_en_curve = df_en_curve.to_csv(index=False).encode('utf-8')
        
        img_en_buf = io.BytesIO()
        fig_en.savefig(img_en_buf, format="png", dpi=300)
        img_en_buf.seek(0)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(label="E-N 곡선 데이터 (CSV) 다운로드", data=csv_en_curve, file_name="en_curve_data.csv", mime="text/csv", use_container_width=True)
        with dl_col2:
            st.download_button(label="E-N 곡선 이미지 (PNG) 다운로드", data=img_en_buf, file_name="en_curve_plot.png", mime="image/png", use_container_width=True)

        with st.expander("E-N 곡선 수치 데이터 보기"):
            if not df_en_curve.empty:
                # 숫자 형식 지정 (예: 소수점 4자리 또는 과학적 표기법)
                st.dataframe(df_en_curve.style.format({
                    col: '{:.4e}' for col in df_en_curve.columns if df_en_curve[col].dtype == 'float'
                }))
            else:
                st.write("표시할 E-N 곡선 데이터가 없습니다.")

    # Tab 3: Shear Curve (Gamma-N) - Conditional (기존 Tab 4)
    # 전단 결과는 항상 계산되므로, st.session_state.shear_results 유무로 표시 결정
    if st.session_state.shear_results:
        with tabs[2]: # 탭 인덱스 수정 (0, 1, 2)
            st.subheader("Gamma-N (전단 변형률-수명) 곡선")
            shear_data = st.session_state.shear_results
            en_base_data = st.session_state.en_results 
            
            fig_gn, ax_gn = plt.subplots(figsize=(10, 6))
            reversals_gn = en_base_data.get("Nf", np.array([])) 

            tauf_prime = shear_data.get('tauf_MPa')
            gammaf_prime = shear_data.get('gammaf')
            b0 = shear_data.get('b0')
            c0 = shear_data.get('c0')
            G_mpa = shear_data.get('G_mpa')

            if all(val is not None and not np.isnan(val) for val in [tauf_prime, gammaf_prime, b0, c0, G_mpa]) and G_mpa > 0:
                elastic_shear_strain_gn = (tauf_prime / G_mpa) * (reversals_gn ** b0)
                plastic_shear_strain_gn = gammaf_prime * (reversals_gn ** c0)
                total_shear_strain_gn = elastic_shear_strain_gn + plastic_shear_strain_gn
                
                ax_gn.loglog(reversals_gn, total_shear_strain_gn, '-', label='Total Shear Strain (γ_a,pred)', linewidth=2)
                ax_gn.loglog(reversals_gn, elastic_shear_strain_gn, '--', label='Elastic Shear Strain (pred)', alpha=0.8)
                ax_gn.loglog(reversals_gn, plastic_shear_strain_gn, ':', label='Plastic Shear Strain (pred)', alpha=0.8)

                ax_gn.set_xlabel('Reversals to Failure (2Nf)')
                ax_gn.set_ylabel('Shear Strain Amplitude (γ_a)')
                ax_gn.legend()
                ax_gn.grid(True, which="both", ls="-", alpha=0.7)
                valid_strains_gn = total_shear_strain_gn[~np.isnan(total_shear_strain_gn)]
                if len(valid_strains_gn) > 0:
                     ax_gn.set_ylim(bottom=max(1e-5, np.min(valid_strains_gn) * 0.5), top=np.max(valid_strains_gn) * 1.5)
                else:
                    ax_gn.set_ylim(bottom=1e-5)
                ax_gn.set_title('Predicted Gamma-N (Shear Strain-Life) Curve')
                st.caption(f"전단 변환 방법: {shear_data.get('conversion_method', 'N/A')}") # 변환 방법은 여기에 유지
            else:
                ax_gn.text(0.5, 0.5, 'Gamma-N Curve data not available or contains NaNs.', ha='center', va='center', transform=ax_gn.transAxes)
                ax_gn.set_title('Predicted Gamma-N (Shear Strain-Life) Curve (Data N/A)')
                st.warning("전단 피로 파라미터 계산에 실패했거나 전단 탄성 계수가 0 또는 NaN이므로 Gamma-N 곡선을 생성할 수 없습니다.")

            # st.latex(r"\\frac{\\Delta\\gamma}{2} = \\frac{\\tau'_f}{G}\\,(2N_f)^{b_0} + \\gamma'_f\\,(2N_f)^{c_0}") # 파라미터 탭으로 이동
            st.pyplot(fig_gn)

            # Download buttons for Gamma-N curve
            # total_shear_strain_gn 등이 계산되지 않았을 수 있으므로 확인 후 DataFrame 생성
            df_gn_data_to_save = {}
            if 'total_shear_strain_gn' in locals() and total_shear_strain_gn is not None:
                df_gn_data_to_save['Total_Shear_Strain_Amplitude'] = total_shear_strain_gn
            if 'elastic_shear_strain_gn' in locals() and elastic_shear_strain_gn is not None:
                df_gn_data_to_save['Elastic_Shear_Strain_Amplitude'] = elastic_shear_strain_gn
            if 'plastic_shear_strain_gn' in locals() and plastic_shear_strain_gn is not None:
                df_gn_data_to_save['Plastic_Shear_Strain_Amplitude'] = plastic_shear_strain_gn
            
            if df_gn_data_to_save: # 데이터가 하나라도 있으면
                df_gn_data_to_save['2Nf'] = reversals_gn # 2Nf는 항상 있음
                df_gn_curve = pd.DataFrame(df_gn_data_to_save)
                # 컬럼 순서 재정의 (2Nf가 맨 앞으로 오도록)
                cols_order = ['2Nf'] + [col for col in df_gn_data_to_save if col != '2Nf']
                df_gn_curve = df_gn_curve[cols_order]
                csv_gn_curve = df_gn_curve.to_csv(index=False).encode('utf-8')
                gn_csv_disabled = False
            else:
                csv_gn_curve = pd.DataFrame().to_csv(index=False).encode('utf-8')
                gn_csv_disabled = True

            img_gn_buf = io.BytesIO()
            fig_gn.savefig(img_gn_buf, format="png", dpi=300)
            img_gn_buf.seek(0)
            
            dl_gn_col1, dl_gn_col2 = st.columns(2)
            with dl_gn_col1:
                st.download_button(label="Gamma-N 곡선 데이터 (CSV) 다운로드", data=csv_gn_curve, file_name="gamma_n_curve_data.csv", mime="text/csv",
                                   disabled=gn_csv_disabled, use_container_width=True)
            with dl_gn_col2:
                st.download_button(label="Gamma-N 곡선 이미지 (PNG) 다운로드", data=img_gn_buf, file_name="gamma_n_curve_plot.png", mime="image/png", use_container_width=True)
            
            with st.expander("Gamma-N 곡선 수치 데이터 보기"):
                # df_gn_curve는 이미 위에서 생성되었거나 빈 DataFrame임
                if not csv_gn_curve == pd.DataFrame().to_csv(index=False).encode('utf-8'): # gn_csv_disabled 대신 csv 내용으로 확인
                    # df_gn_curve를 다시 만들거나, 이미 있는 df_gn_curve를 사용
                    # 위에서 df_gn_curve가 비어있을 수 있으므로, 여기서 다시 만들거나, 안전하게 접근
                    if 'df_gn_curve' in locals() and not df_gn_curve.empty:
                         st.dataframe(df_gn_curve.style.format({
                            col: '{:.4e}' for col in df_gn_curve.columns if df_gn_curve[col].dtype == 'float'
                        }))
                    elif df_gn_data_to_save: # df_gn_data_to_save로 DataFrame을 만들 수 있다면
                        temp_df_gn = pd.DataFrame(df_gn_data_to_save)
                        cols_order_temp = ['2Nf'] + [col for col in df_gn_data_to_save if col != '2Nf']
                        temp_df_gn = temp_df_gn[cols_order_temp]
                        st.dataframe(temp_df_gn.style.format({
                            col: '{:.4e}' for col in temp_df_gn.columns if temp_df_gn[col].dtype == 'float'
                        }))
                    else:
                        st.write("표시할 Gamma-N 곡선 데이터가 없습니다 (계산 실패 또는 데이터 없음).")
                else:
                    st.write("표시할 Gamma-N 곡선 데이터가 없습니다.")
    
    # 만약 shear_results가 없으면 (예외 발생 등) 전단 탭을 표시하지 않거나, 오류 메시지만 표시할 수 있음
    # 현재 로직에서는 tabs[2]가 항상 전단 탭이므로, shear_results가 없을 경우 위에서 내용이 안그려짐

elif not st.session_state.prediction_triggered:
    st.info('👈 사이드바에서 재료 물성치를 입력하고, 예측 버튼을 누르세요.') # 문구 수정됨

st.divider() # 시각적 구분을 위한 선 추가
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 0.8em;">
        © 2025 YeoJoon Yoon. All Rights Reserved.<br>
        Contact: <a href="mailto:goat@sogang.ac.kr">goat@sogang.ac.kr</a>
    </div>
    """,
    unsafe_allow_html=True
)

# 앱 실행 안내 (터미널에서 직접 실행 시 필요)
# streamlit run FatiguePredictor0528.py
