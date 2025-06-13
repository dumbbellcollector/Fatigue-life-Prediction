import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calculate_average_composition(row, element):
    """
    각 원소의 Min, Max 값의 평균을 계산
    비어있는 셀은 0으로 처리
    """
    min_col = f"{element}_Min"
    max_col = f"{element}_Max"
    
    min_val = row[min_col] if pd.notna(row[min_col]) else 0
    max_val = row[max_col] if pd.notna(row[max_col]) else 0
    
    return (min_val + max_val) / 2

def classify_alloy_type(composition_row):
    """
    ISO 4948 기준으로 합금 타입 분류
    
    Parameters:
    composition_row: pandas Series with element composition in wt%
    
    Returns:
    str: 'unalloyed', 'lowalloy', 'highalloy'
    """
    
    # Unalloyed Steel 기준 체크
    unalloyed_criteria = {
        'Mn': 1.65,  # 망간 1.65% 이하
        'Si': 0.60,  # 실리콘 0.60% 이하  
        'Cu': 0.60,  # 구리 0.60% 이하
    }
    
    # 기타 합금 원소 (크롬, 니켈, 몰리브덴 등) 0.40% 이하
    other_elements = ['Cr', 'Ni', 'Mo', 'V', 'Nb', 'Ti', 'Al']
    
    # 탄소 함량 체크 (2% 이하)
    carbon_content = composition_row['C']
    if carbon_content > 2.0:
        return 'highalloy'  # 탄소 함량이 2% 초과면 고합금강
    
    # Unalloyed Steel 기준 체크
    is_unalloyed = True
    
    # 망간, 실리콘, 구리 기준 체크
    for element, limit in unalloyed_criteria.items():
        if composition_row[element] > limit:
            is_unalloyed = False
            break
    
    # 기타 합금 원소 기준 체크 (각각 0.40% 이하)
    if is_unalloyed:
        for element in other_elements:
            if composition_row[element] > 0.40:
                is_unalloyed = False
                break
    
    if is_unalloyed:
        return 'unalloyed'
    
    # 합금 원소 총합 계산 (탄소 제외)
    # 주요 합금 원소들의 총합
    alloy_elements = ['Mn', 'Si', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Nb', 'Ti', 'Al']
    total_alloy_content = sum(composition_row[element] for element in alloy_elements)
    
    # Low-alloy vs High-alloy 구분
    if total_alloy_content <= 5.0:
        return 'lowalloy'
    else:
        return 'highalloy'

def main():
    # 데이터 파일 경로
    data_path = "TrainSet0507_NoDuplicatesHV.xlsx"
    output_path = "TrainSet0507_CompositionClassification.xlsx"
    
    print("Excel 파일 읽는 중...")
    # composition 시트 읽기
    df = pd.read_excel(data_path, sheet_name='composition')
    print(f"총 {len(df)}개 샘플 로드됨")
    
    # 원소 목록 (Min, Max 컬럼이 있는 원소들)
    elements = ['C', 'Mn', 'P', 'S', 'Si', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'B', 'Al', 'N', 'Nb', 'Ti']
    
    print("원소별 평균 성분 계산 중...")
    # 각 원소별로 평균 성분 계산
    for element in elements:
        df[element] = df.apply(lambda row: calculate_average_composition(row, element), axis=1)
        print(f"  {element}: 평균 = {df[element].mean():.4f}%, 최대 = {df[element].max():.4f}%")
    
    print("\nISO 4948 기준으로 합금 분류 중...")
    # 각 샘플에 대해 합금 타입 분류
    df['alloy_type'] = df.apply(lambda row: classify_alloy_type(row), axis=1)
    
    # 분류 결과 통계
    classification_counts = df['alloy_type'].value_counts()
    print("\n=== 분류 결과 ===")
    for alloy_type, count in classification_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{alloy_type}: {count}개 샘플 ({percentage:.1f}%)")
    
    print(f"\n분류된 데이터를 {output_path}에 저장 중...")
    # ExcelWriter를 사용하여 여러 시트로 저장
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 각 합금 타입별로 시트 생성
        for alloy_type in ['unalloyed', 'lowalloy', 'highalloy']:
            subset_df = df[df['alloy_type'] == alloy_type].copy()
            
            if len(subset_df) > 0:
                # alloy_type 컬럼 제거 (분류용이므로)
                subset_df = subset_df.drop('alloy_type', axis=1)
                subset_df.to_excel(writer, sheet_name=alloy_type, index=False)
                print(f"  '{alloy_type}' 시트: {len(subset_df)}개 샘플 저장")
            else:
                print(f"  '{alloy_type}' 시트: 해당하는 샘플이 없음")
    
    print(f"\n✅ 분류 완료! 파일이 {output_path}에 저장되었습니다.")
    
    # 각 그룹의 대표적인 성분 정보 출력
    print("\n=== 각 그룹의 대표적인 성분 정보 ===")
    for alloy_type in ['unalloyed', 'lowalloy', 'highalloy']:
        subset_df = df[df['alloy_type'] == alloy_type]
        if len(subset_df) > 0:
            print(f"\n{alloy_type.upper()} 그룹 ({len(subset_df)}개 샘플):")
            key_elements = ['C', 'Mn', 'Si', 'Ni', 'Cr', 'Mo']
            for element in key_elements:
                mean_val = subset_df[element].mean()
                max_val = subset_df[element].max()
                print(f"  {element}: 평균 {mean_val:.3f}%, 최대 {max_val:.3f}%")
            
            # 총 합금 원소 함량
            alloy_elements = ['Mn', 'Si', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Nb', 'Ti', 'Al']
            total_alloy = subset_df[alloy_elements].sum(axis=1)
            print(f"  총 합금 원소: 평균 {total_alloy.mean():.3f}%, 최대 {total_alloy.max():.3f}%")

if __name__ == "__main__":
    main()
