# --- START OF FILE composition_to_properties.py ---
import numpy as np

def calculate_e_gpa(composition_wt_percent: dict) -> float:
    """
    Calculates Elastic Modulus (E) in GPa based on composition using improved regression.
    [2groups], 원소: C, 기준: C ≤ 0.3500, 모델: Power/Power, adj_R^2: 0.2699
    """
    C = composition_wt_percent.get('C', 0)
    Cr = composition_wt_percent.get('Cr', 0)
    Si = composition_wt_percent.get('Si', 0)
    S = composition_wt_percent.get('S', 0)
    
    try:
        if C <= 0.3500:
            # [low 그룹] E = 207.585408 * C^0.067282 * Cr^-0.020176 * Si^-0.055693
            # Handle zero values for power operations
            C_term = max(C, 1e-6) ** 0.067282 if C > 0 else 1.0
            Cr_term = max(Cr, 1e-6) ** (-0.020176) if Cr > 0 else 1.0
            Si_term = max(Si, 1e-6) ** (-0.055693) if Si > 0 else 1.0
            e_gpa = 207.585408 * C_term * Cr_term * Si_term
        else:
            # [high 그룹] E = 221.743765 * S^0.023998
            S_term = max(S, 1e-6) ** 0.023998 if S > 0 else 1.0
            e_gpa = 221.743765 * S_term
        return max(e_gpa, 1.0)  # Ensure positive value
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def calculate_ys_mpa(composition_wt_percent: dict) -> float:
    """
    Calculates Yield Strength (YS) in MPa based on composition using improved regression.
    [2groups], 원소: Mn, 기준: Mn ≤ 0.5500, 모델: Linear/Exp, adj_R^2: 0.7000
    """
    C = composition_wt_percent.get('C', 0)
    Mn = composition_wt_percent.get('Mn', 0)
    Mo = composition_wt_percent.get('Mo', 0)
    P = composition_wt_percent.get('P', 0)
    S = composition_wt_percent.get('S', 0)
    
    try:
        if Mn <= 0.5500:
            # [low 그룹] YS = 999.492124 + 1691.791912*C + -1110.803461*Mn + -11979.529905*P
            ys_mpa = 999.492124 + 1691.791912*C - 1110.803461*Mn - 11979.529905*P
        else:
            # [high 그룹] YS = exp(7.225424) * exp(C*1.685394) * exp(Mo*-0.834706) * exp(P*-60.401816) * exp(S*29.055291)
            ys_mpa = (np.exp(7.225424) * np.exp(C*1.685394) * np.exp(Mo*(-0.834706)) * 
                     np.exp(P*(-60.401816)) * np.exp(S*29.055291))
        return max(ys_mpa, 1.0)  # Ensure positive value
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def calculate_ts_mpa(composition_wt_percent: dict) -> float:
    """
    Calculates Tensile Strength (TS) in MPa based on composition using improved regression.
    [3groups], 원소: Mn, 기준: Mn ≤ 0.5500, 0.5500 < Mn ≤ 1.2500, Mn > 1.2500, 모델: Power/Exp/Exp, adj_R^2: 0.7940
    """
    C = composition_wt_percent.get('C', 0)
    Mn = composition_wt_percent.get('Mn', 0)
    Mo = composition_wt_percent.get('Mo', 0)
    P = composition_wt_percent.get('P', 0)
    S = composition_wt_percent.get('S', 0)
    
    try:
        if Mn <= 0.5500:
            # [low 그룹] TS = 729.127305 * C^0.798953 * Mn^-0.749087 * S^-0.140238
            C_term = max(C, 1e-6) ** 0.798953 if C > 0 else 1.0
            Mn_term = max(Mn, 1e-6) ** (-0.749087) if Mn > 0 else 1.0
            S_term = max(S, 1e-6) ** (-0.140238) if S > 0 else 1.0
            ts_mpa = 729.127305 * C_term * Mn_term * S_term
        elif Mn <= 1.2500:
            # [mid 그룹] TS = exp(7.145395) * exp(C*1.045808) * exp(Mo*-0.738198) * exp(P*-44.508794) * exp(S*31.846462)
            ts_mpa = (np.exp(7.145395) * np.exp(C*1.045808) * np.exp(Mo*(-0.738198)) * 
                     np.exp(P*(-44.508794)) * np.exp(S*31.846462))
        else:  # Mn > 1.2500
            # [high 그룹] TS = exp(5.451138) * exp(C*2.019657) * exp(Mn*0.302011)
            ts_mpa = np.exp(5.451138) * np.exp(C*2.019657) * np.exp(Mn*0.302011)
        return max(ts_mpa, 1.0)  # Ensure positive value
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def calculate_hb(composition_wt_percent: dict) -> float:
    """
    Calculates Brinell Hardness (HB) based on composition using improved regression.
    [2groups], 원소: C, 기준: C ≤ 0.3000, 모델: Power/Exp, adj_R^2: 0.5902
    """
    C = composition_wt_percent.get('C', 0)
    Cr = composition_wt_percent.get('Cr', 0)
    Mn = composition_wt_percent.get('Mn', 0)
    P = composition_wt_percent.get('P', 0)
    S = composition_wt_percent.get('S', 0)
    
    try:
        if C <= 0.3000:
            # [low 그룹] HB = 16.771793 * C^0.551087 * P^-1.008093
            C_term = max(C, 1e-6) ** 0.551087 if C > 0 else 1.0
            P_term = max(P, 1e-6) ** (-1.008093) if P > 0 else 1.0
            hb = 16.771793 * C_term * P_term
        else:
            # [high 그룹] HB = exp(4.899457) * exp(Cr*0.206266) * exp(Mn*1.013949) * exp(P*-10.154118) * exp(S*7.290371)
            hb = (np.exp(4.899457) * np.exp(Cr*0.206266) * np.exp(Mn*1.013949) * 
                 np.exp(P*(-10.154118)) * np.exp(S*7.290371))
        return max(hb, 1.0)  # Ensure positive value
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

def calculate_monotonic_properties(composition_wt_percent: dict) -> dict:
    """
    Calculates all monotonic properties (E_GPa, YS_MPa, TS_MPa, HB)
    from the given alloy composition (wt%) using improved group-based regression models.

    Args:
        composition_wt_percent (dict): A dictionary where keys are element symbols
                                       (e.g., 'C', 'Mn', 'Cr', 'Si', 'S', 'Mo', 'P') and values are their
                                       weight percentages.
                                       Missing elements will be treated as 0 wt%.

    Returns:
        dict: A dictionary containing the calculated 'E_gpa', 'YS_mpa', 'TS_mpa', 'HB'.
              Returns NaN for a property if calculation fails.
    """
    e_gpa = calculate_e_gpa(composition_wt_percent)
    ys_mpa = calculate_ys_mpa(composition_wt_percent)
    ts_mpa = calculate_ts_mpa(composition_wt_percent)
    hb = calculate_hb(composition_wt_percent)

    # 계산된 물성치가 물리적으로 타당한지 간단히 확인
    properties = {
        'E_gpa': e_gpa if not np.isnan(e_gpa) and e_gpa > 0 else np.nan,
        'YS_mpa': ys_mpa if not np.isnan(ys_mpa) and ys_mpa > 0 else np.nan,
        'TS_mpa': ts_mpa if not np.isnan(ts_mpa) and ts_mpa > 0 else np.nan,
        'HB': hb if not np.isnan(hb) and hb > 0 else np.nan
    }
    return properties

if __name__ == '__main__':
    # Example Usage and Unit Test with improved models
    sample_composition = {
        'C': 0.2,
        'Mn': 0.8,
        'Si': 0.3,
        'Cr': 1.0,
        'Mo': 0.2,
        'P': 0.01,
        'S': 0.01
    }
    
    print(f"Sample Composition (wt%): {sample_composition}")
    calculated_props = calculate_monotonic_properties(sample_composition)
    print("\nCalculated Monotonic Properties (Improved Models):")
    if calculated_props:
        print(f"  E (GPa): {calculated_props.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props.get('HB', 'N/A'):.2f}")

    # Test with different carbon content groups
    low_carbon_composition = {'C': 0.1, 'Mn': 0.3, 'Si': 0.2, 'P': 0.015, 'S': 0.015}
    print(f"\nLow Carbon Composition (wt%): {low_carbon_composition}")
    calculated_props_low_c = calculate_monotonic_properties(low_carbon_composition)
    print("Calculated Properties (Low Carbon):")
    if calculated_props_low_c:
        print(f"  E (GPa): {calculated_props_low_c.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props_low_c.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props_low_c.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props_low_c.get('HB', 'N/A'):.2f}")

    # Test with high manganese content
    high_mn_composition = {'C': 0.3, 'Mn': 1.5, 'Cr': 0.5, 'Mo': 0.1, 'P': 0.02, 'S': 0.02}
    print(f"\nHigh Mn Composition (wt%): {high_mn_composition}")
    calculated_props_high_mn = calculate_monotonic_properties(high_mn_composition)
    print("Calculated Properties (High Mn):")
    if calculated_props_high_mn:
        print(f"  E (GPa): {calculated_props_high_mn.get('E_gpa', 'N/A'):.2f}")
        print(f"  YS (MPa): {calculated_props_high_mn.get('YS_mpa', 'N/A'):.2f}")
        print(f"  TS (MPa): {calculated_props_high_mn.get('TS_mpa', 'N/A'):.2f}")
        print(f"  HB: {calculated_props_high_mn.get('HB', 'N/A'):.2f}")

    # Test edge cases
    print("\n=== Testing Group Boundaries ===")
    
    # C boundary test (0.3000 for HB, 0.3500 for E)
    c_boundary_composition = {'C': 0.3001, 'Mn': 0.6, 'Si': 0.25, 'Cr': 0.2, 'P': 0.02, 'S': 0.02}
    print(f"C Boundary Test (C=0.3001): {c_boundary_composition}")
    props_c_boundary = calculate_monotonic_properties(c_boundary_composition)
    print(f"  E (GPa): {props_c_boundary.get('E_gpa', 'N/A'):.2f}")
    print(f"  HB: {props_c_boundary.get('HB', 'N/A'):.2f}")
    
    # Mn boundary test (0.5500, 1.2500)
    mn_mid_composition = {'C': 0.25, 'Mn': 0.8, 'Mo': 0.15, 'P': 0.015, 'S': 0.015}
    print(f"Mn Mid-Range Test (Mn=0.8): {mn_mid_composition}")
    props_mn_mid = calculate_monotonic_properties(mn_mid_composition)
    print(f"  YS (MPa): {props_mn_mid.get('YS_mpa', 'N/A'):.2f}")
    print(f"  TS (MPa): {props_mn_mid.get('TS_mpa', 'N/A'):.2f}")

# --- END OF FILE composition_to_properties.py ---