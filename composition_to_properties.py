# --- START OF FILE composition_to_properties.py ---
import numpy as np

# 각 물성치 계산을 위한 선형 회귀식 계수 (가독성을 위해 딕셔너리로 관리)
# 계수 순서: Intercept, Cr, Mo, C, Mn, P, S (없는 경우 0 또는 해당 원소 제외)
# 참고: E는 P, S, C, Mn의 계수가 없으므로 0으로 처리하거나, 함수 호출 시 해당 원소를 전달하지 않음.

PROPERTY_COEFFICIENTS = {
    'E_GPa': { # E는 GPa 단위로 직접 계산
        'intercept': 206.8566197777149,
        'Cr': 4.181685172506121,
        'Mo': -31.120581999426335,
        # C, Mn, P, S 계수는 0 또는 생략 (수식에 없음)
    },
    'YS_MPa': {
        'intercept': 873.6097341379337,
        'C': 1140.1801429332056,
        'Mn': 409.5664199265409,
        'P': -54238.0933827761,
        'S': 26114.7294747601,
        'Cr': 346.9667147626457,
        'Mo': -863.7860530291761,
    },
    'TS_MPa': {
        'intercept': 886.2510722103376,
        'C': 1203.7132537819582,
        'Mn': 492.13650249092916,
        'P': -61364.753186154434,
        'S': 37142.54415129786,
        'Cr': 360.8312658083453,
        'Mo': -900.0008381153384,
    },
    'HB': {
        'intercept': 241.66243906689817,
        'C': 269.54357276094794,
        'Mn': 126.55854659902312,
        'P': -15488.73440473908,
        'S': 9570.48827318713,
        'Cr': 97.60694801374052,
        'Mo': -208.21673893966522,
    }
}

# 계산에 사용될 원소 리스트 (수식에 있는 모든 원소)
ALLOY_ELEMENTS_FOR_CALC = ['Cr', 'Mo', 'C', 'Mn', 'P', 'S']

def calculate_e_gpa(composition_wt_percent: dict) -> float:
    """Calculates Elastic Modulus (E) in GPa based on composition."""
    coeffs = PROPERTY_COEFFICIENTS['E_GPa']
    e_gpa = coeffs.get('intercept', 0)
    # E 수식에 포함된 원소만 사용
    e_gpa += coeffs.get('Cr', 0) * composition_wt_percent.get('Cr', 0)
    e_gpa += coeffs.get('Mo', 0) * composition_wt_percent.get('Mo', 0)
    return e_gpa

def calculate_ys_mpa(composition_wt_percent: dict) -> float:
    """Calculates Yield Strength (YS) in MPa based on composition."""
    coeffs = PROPERTY_COEFFICIENTS['YS_MPa']
    ys_mpa = coeffs.get('intercept', 0)
    for element in ['C', 'Mn', 'P', 'S', 'Cr', 'Mo']: # YS 수식에 포함된 원소 순서대로
        ys_mpa += coeffs.get(element, 0) * composition_wt_percent.get(element, 0)
    return ys_mpa

def calculate_ts_mpa(composition_wt_percent: dict) -> float:
    """Calculates Tensile Strength (TS) in MPa based on composition."""
    coeffs = PROPERTY_COEFFICIENTS['TS_MPa']
    ts_mpa = coeffs.get('intercept', 0)
    for element in ['C', 'Mn', 'P', 'S', 'Cr', 'Mo']: # TS 수식에 포함된 원소 순서대로
        ts_mpa += coeffs.get(element, 0) * composition_wt_percent.get(element, 0)
    return ts_mpa

def calculate_hb(composition_wt_percent: dict) -> float:
    """Calculates Brinell Hardness (HB) based on composition."""
    coeffs = PROPERTY_COEFFICIENTS['HB']
    hb = coeffs.get('intercept', 0)
    for element in ['C', 'Mn', 'P', 'S', 'Cr', 'Mo']: # HB 수식에 포함된 원소 순서대로
        hb += coeffs.get(element, 0) * composition_wt_percent.get(element, 0)
    return hb

def calculate_monotonic_properties(composition_wt_percent: dict) -> dict:
    """
    Calculates all monotonic properties (E_GPa, YS_MPa, TS_MPa, HB)
    from the given alloy composition (wt%).

    Args:
        composition_wt_percent (dict): A dictionary where keys are element symbols
                                       (e.g., 'C', 'Mn', 'Cr') and values are their
                                       weight percentages.
                                       Expected elements: Cr, Mo, C, Mn, P, S.
                                       Missing elements will be treated as 0 wt%.

    Returns:
        dict: A dictionary containing the calculated 'E_gpa', 'YS_mpa', 'TS_mpa', 'HB'.
              Returns None for a property if calculation is not possible (e.g. negative prediction).
    """
    # 입력된 딕셔너리에 모든 ALLOY_ELEMENTS_FOR_CALC가 있는지 확인하고, 없으면 0으로 채움
    # (calculate_e_gpa 등 내부 함수에서 .get(element, 0)으로 이미 처리하고 있으므로 중복일 수 있으나 명시적)
    full_composition = {el: composition_wt_percent.get(el, 0) for el in ALLOY_ELEMENTS_FOR_CALC}

    e_gpa = calculate_e_gpa(full_composition)
    ys_mpa = calculate_ys_mpa(full_composition)
    ts_mpa = calculate_ts_mpa(full_composition)
    hb = calculate_hb(full_composition)

    # 계산된 물성치가 물리적으로 타당한지 간단히 확인 (음수 값 방지 등)
    # YS > TS 는 예측 단계에서 경고 처리
    properties = {
        'E_gpa': e_gpa if e_gpa > 0 else np.nan, # E는 0보다 커야 함
        'YS_mpa': ys_mpa if ys_mpa > 0 else np.nan, # YS는 0보다 커야 함
        'TS_mpa': ts_mpa if ts_mpa > 0 else np.nan, # TS는 0보다 커야 함
        'HB': hb if hb > 0 else np.nan # HB는 0보다 커야 함 (또는 0 허용 시 hb >= 0)
    }
    return properties

if __name__ == '__main__':
    # Example Usage and Unit Test
    sample_composition = {
        'C': 0.2,
        'Mn': 0.8,
        'Si': 0.3, # Si는 현재 수식에 없으므로 무시됨 (get으로 처리)
        'Cr': 1.0,
        'Mo': 0.2,
        'P': 0.01,
        'S': 0.01
    }
    
    print(f"Sample Composition (wt%): {sample_composition}")
    calculated_props = calculate_monotonic_properties(sample_composition)
    print("\nCalculated Monotonic Properties:")
    if calculated_props:
        print(f"  E (GPa): {calculated_props.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props.get('HB', 'N/A'):.2f}")

    # Test with extreme or missing values
    edge_case_composition = {'C': 0.1, 'P': 0.1} # 다른 값은 0으로 처리
    print(f"\nEdge Case Composition (wt%): {edge_case_composition}")
    calculated_props_edge = calculate_monotonic_properties(edge_case_composition)
    print("\nCalculated Monotonic Properties (Edge Case):")
    if calculated_props_edge:
        print(f"  E (GPa): {calculated_props_edge.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props_edge.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props_edge.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props_edge.get('HB', 'N/A'):.2f}")

    # Test with potentially problematic P and S values (leading to negative YS/TS/HB)
    problem_composition = {'P': 0.05, 'S': 0.05} # P, S 계수가 커서 음수 유발 가능성 테스트
    print(f"\nProblematic Composition (wt%): {problem_composition}")
    calculated_props_problem = calculate_monotonic_properties(problem_composition)
    print("\nCalculated Monotonic Properties (Problematic Case):")
    if calculated_props_problem:
        print(f"  E (GPa): {calculated_props_problem.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props_problem.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props_problem.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props_problem.get('HB', 'N/A'):.2f}")
# --- END OF FILE composition_to_properties.py ---