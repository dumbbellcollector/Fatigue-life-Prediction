# Cell 12: GUI 고려사항 (Streamlit 예제)

# 이 셀은 Streamlit 앱의 기본 구조를 제공합니다.
# 실행하려면 코드를 .py 파일(예: app.py)로 저장하고 터미널에서 `streamlit run app.py`를 실행하세요.
# 훈련된 모델(.pth)과 스케일러(.pkl)가 동일한 디렉토리에 있는지 확인하거나 올바른 경로를 제공해야 합니다.
# 또한 predict_fatigue_params_and_curve 함수가 앱 스크립트에서 정의되거나 가져와야 합니다.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # 모델 정의 필요
import joblib
import os # 파일 존재 확인용
import pandas as pd

# --- 모델 정의 복제 ---
# 모델 정의를 별도의 .py 파일에 두고 가져오는 것이 좋은 방법입니다.
# 여기서는 간단하게 재정의하거나 사용 가능하다고 가정합니다.
class FatiguePINN(nn.Module): # Cell 7에서 정확한 정의 복사
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128]): # 동일한 아키텍처 사용
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

# --- 예측 함수 복제 ---
# 또는 유틸리티 파일에서 가져오기
def predict_fatigue_params_and_curve(E_val, YS_val, TS_val, HB_val, model, scaler_X, scaler_y, device, mode='tensile', nu=0.3):
    # Cell 11에서 정확한 함수 복사
    model.eval()
    input_features = np.array([[E_val, YS_val, TS_val, HB_val]])
    input_scaled = scaler_X.transform(input_features)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        predicted_params_scaled = model(input_tensor)
    predicted_params_orig = scaler_y.inverse_transform(predicted_params_scaled.cpu().numpy())[0]
    target_cols = ['spf_MPa', 'b', 'epf', 'c'] # 훈련과 일치하는지 확인
    predicted_params_dict = {name: val for name, val in zip(target_cols, predicted_params_orig)}
    
    if mode.lower() == 'tensile':
        sigma_f_prime = predicted_params_dict['spf_MPa']
        b = predicted_params_dict['b']
        epsilon_f_prime = predicted_params_dict['epf']
        c = predicted_params_dict['c']
        reversals = np.logspace(1, 7, num=100)
        # 0으로 나누기 또는 잘못된 E_val 방지
        if E_val <= 0: E_val = 1e-6 # 0으로 나누기 방지, 더 나은 처리 방법 고려
        elastic_strain = (sigma_f_prime / E_val) * (reversals ** b)
        plastic_strain = epsilon_f_prime * (reversals ** c)
        total_strain_amplitude = elastic_strain + plastic_strain
        return predicted_params_dict, reversals, total_strain_amplitude
    
    elif mode.lower() == 'shear':
        # 전단 모드 변환 (von Mises 또는 Tresca 기준 사용)
        conversion_method = "von Mises"  # 또는 "Tresca"
        
        # 인장 파라미터 추출
        sigma_f_prime = predicted_params_dict['spf_MPa']
        b = predicted_params_dict['b']
        epsilon_f_prime = predicted_params_dict['epf']
        c = predicted_params_dict['c']
        
        # 전단 파라미터 계산 (von Mises)
        tau_f_prime = sigma_f_prime / np.sqrt(3)
        b0 = b  # 일반적으로 동일하게 유지
        gamma_f_prime = epsilon_f_prime * np.sqrt(3)
        c0 = c  # 일반적으로 동일하게 유지
        
        # 전단 탄성 계수 계산
        G = E_val / (2 * (1 + nu))
        
        # 결과 저장
        shear_params = {
            'tauf_MPa': tau_f_prime,
            'b0': b0,
            'gammaf': gamma_f_prime,
            'c0': c0,
            'conversion_method': conversion_method
        }
        
        # 전단 변형률 진폭 계산
        reversals = np.logspace(1, 7, num=100)
        elastic_shear_strain = (tau_f_prime / G) * (reversals ** b0)
        plastic_shear_strain = gamma_f_prime * (reversals ** c0)
        total_shear_strain = elastic_shear_strain + plastic_shear_strain
        
        return shear_params, reversals, total_shear_strain


# --- 모델 및 스케일러 로드 ---
@st.cache_resource # 리소스 로딩 캐싱
def load_resources(model_path='best_fatigue_pinn_model.pth', scaler_x_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # 파일 존재 확인
    if not os.path.exists(model_path) or not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        st.error("오류: 모델 또는 스케일러 파일을 찾을 수 없습니다. 올바른 디렉토리에 있는지 확인하세요.")
        return None, None, None, None

    try:
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        # 스케일러에서 모델 입력/출력 차원 결정
        input_dim = scaler_X.n_features_in_
        output_dim = scaler_y.n_features_in_

        # 올바른 차원으로 모델 인스턴스화
        model = FatiguePINN(input_dim, output_dim).to(device) # 가능하면/필요하면 저장된 아키텍처의 hidden_dims 사용
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return model, scaler_X, scaler_y, device
    except Exception as e:
        st.error(f"리소스 로딩 오류: {e}")
        return None, None, None, None

# --- Streamlit 앱 레이아웃 ---
st.set_page_config(layout="wide")
st.title('금속 재료 피로 수명(E-N 곡선) 예측기')
st.write("PINN 모델을 사용하여 인장 피로 매개변수와 E-N 곡선을 예측하기 위해 재료 속성을 입력하세요.")

# 리소스 로드
model, scaler_X, scaler_y, device = load_resources()

if model is None: # 리소스 로드 실패 시 중지
    st.stop()


# 입력 섹션
col1, col2 = st.columns([1, 2])

with col1:
    st.header("재료 속성 입력")
    e_mod = st.number_input('탄성 계수(E, MPa)', min_value=1.0, value=200000.0, format='%.1f')
    ys = st.number_input('항복 강도(YS, MPa)', min_value=1.0, value=500.0, format='%.1f')
    ts = st.number_input('인장 강도(UTS, MPa)', min_value=1.0, value=700.0, format='%.1f')
    hb_input = st.number_input('브리넬 경도(HB)', min_value=0.0, value=200.0, format='%.1f', help="알 수 없는 경우 0을 입력하거나 비워두세요(모델이 대체를 시도합니다).")
    nu_val = st.number_input('포아송 비(ν)', min_value=0.0, max_value=0.5, value=0.3, format='%.2f')
    
    # 알 수 없는 HB 처리 - 기본 전략: 0인 경우 훈련 데이터 스케일러의 평균 HB 사용
    # 더 강력한 접근 방식은 HB가 0인 경우 저장된 HB 대체 모델을 사용합니다.
    hb_processed = hb_input
    if hb_input == 0.0:
         try:
            # HB가 scaler_X에 의해 스케일링된 마지막 특성이라고 가정
            hb_mean_from_scaler = scaler_X.mean_[-1]
            hb_processed = hb_mean_from_scaler
            st.info(f"HB가 제공되지 않았습니다. 훈련 데이터의 평균 HB 사용: {hb_processed:.1f}")
         except Exception:
             st.warning("평균 HB를 검색할 수 없습니다. 입력 값 0을 사용합니다. 예측이 부정확할 수 있습니다.")
             hb_processed = 0.1 # 모델에 필요한 경우 작은 0이 아닌 값 사용
    
    mode = st.radio("예측 모드 선택:", ["인장(Tensile)", "전단(Shear)"], index=0)
    predict_mode = 'tensile' if mode == "인장(Tensile)" else 'shear'
    
    predict_button = st.button('E-N 곡선 예측')

# 출력 섹션
with col2:
    st.header("예측 결과")
    if predict_button:
        if e_mod <= 0 or ys <= 0 or ts <= 0:
            st.error("E, YS 및 TS에 유효한 양수 값을 입력하세요.")
        elif ys > ts * 1.05: # YS가 TS보다 약간 높은 것 허용
            st.warning("항복 강도(YS)가 인장 강도(UTS)에 비해 높아 보입니다. 확인하세요.")
        else:
            try:
                predicted_params, reversals, strain_amplitude = predict_fatigue_params_and_curve(
                    e_mod, ys, ts, hb_processed, model, scaler_X, scaler_y, device, mode=predict_mode, nu=nu_val
                )

                st.subheader("예측된 피로 매개변수:")
                param_df = pd.DataFrame(list(predicted_params.items()), columns=['매개변수', '값'])
                st.dataframe(param_df.style.format({'값': '{:.4f}'}))

                if predict_mode == 'tensile':
                    curve_title = "예측된 E-N 곡선:"
                    y_label = '변형률 진폭 (epsilon_a)'
                    total_label = '예측된 총 변형률'
                    elastic_label = '탄성 변형률'
                    plastic_label = '소성 변형률'
                    sigma_f_prime_pred = predicted_params['spf_MPa']
                    b_pred = predicted_params['b']
                    epsilon_f_prime_pred = predicted_params['epf']
                    c_pred = predicted_params['c']
                    elastic_component = (sigma_f_prime_pred / e_mod) * (reversals ** b_pred)
                    plastic_component = epsilon_f_prime_pred * (reversals ** c_pred)
                else:  # shear mode
                    curve_title = "예측된 Gamma-N 곡선:"
                    y_label = '전단 변형률 진폭 (gamma_a)'
                    total_label = '예측된 총 전단 변형률'
                    elastic_label = '전단 탄성 변형률'
                    plastic_label = '전단 소성 변형률'
                    G_val = e_mod / (2 * (1 + nu_val))
                    tauf_prime = predicted_params['tauf_MPa']
                    b0 = predicted_params['b0']
                    gammaf_prime = predicted_params['gammaf']
                    c0 = predicted_params['c0']
                    elastic_component = (tauf_prime / G_val) * (reversals ** b0)
                    plastic_component = gammaf_prime * (reversals ** c0)

                st.subheader(curve_title)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.loglog(reversals, strain_amplitude, label=total_label)

                # 구성 요소 플롯
                ax.loglog(reversals, elastic_component, '--', label=elastic_label, alpha=0.7)
                ax.loglog(reversals, plastic_component, ':', label=plastic_label, alpha=0.7)

                ax.set_xlabel('파괴까지의 반복 횟수 (2Nf)')
                ax.set_ylabel(y_label)
                ax.set_title(curve_title.replace(':', ''))
                ax.legend()
                ax.grid(True, which="both", ls="--")
                ax.set_ylim(bottom=max(1e-5, min(strain_amplitude)*0.5), top=max(strain_amplitude)*1.2) # 동적 y 한계
                st.pyplot(fig)

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
                # 디버깅을 위해 선택적으로 더 자세한 트레이스백 출력
                # import traceback
                # st.text(traceback.format_exc())

    else:
        st.info("재료 속성을 입력하고 'E-N 곡선 예측'을 클릭하세요.")



# 이 Streamlit 앱을 실행하려면:
# 1. 위 코드를 (예를 들어) `fatigue_app.py`라는 파일에 저장하세요.
# 2. 'best_fatigue_pinn_model.pth', 'scaler_X.pkl', 'scaler_y.pkl'이 같은 디렉토리에 있는지 확인하세요.
# 3. 터미널을 열고 해당 디렉토리로 이동하세요.
# 4. 다음 명령을 실행하세요: streamlit run fatigue_app.py

print("\nStreamlit 앱 코드가 생성되었습니다. .py 파일로 저장하고 `streamlit run <파일명>.py`로 실행하세요")
# print(streamlit_code) # streamlit run app0.1.py
