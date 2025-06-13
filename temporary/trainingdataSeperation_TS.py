import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def classify_and_save_by_ts(data_path, output_path):
    """
    Excel 파일의 'tensile' 시트를 읽어 TS 기준으로 데이터를 분류하고,
    각 그룹을 별도의 시트로 저장합니다.
    """
    print(f"'{data_path}'에서 데이터 로딩 중...")
    try:
        # 'tensile' 시트만 읽기
        df = pd.read_excel(data_path, sheet_name='Tensile')
        print(f"총 {len(df)}개의 샘플을 'tensile' 시트에서 로드했습니다.")
    except Exception as e:
        print(f"파일 또는 시트를 읽는 중 오류 발생: {e}")
        return

    # TS 컬럼 존재 여부 확인
    if 'TS' not in df.columns:
        print("오류: 'tensile' 시트에 'TS' 컬럼이 없습니다.")
        return

    # TS 값에 따라 그룹 분류
    print("\nTS 값을 기준으로 데이터 분류 중...")
    low_ts_df = df[df['TS'] < 750].copy()
    mid_ts_df = df[(df['TS'] >= 750) & (df['TS'] < 1030)].copy()
    high_ts_df = df[df['TS'] >= 1030].copy()

    # 분류 결과 통계 출력
    print("\n=== 분류 결과 ===")
    print(f"Low TS 그룹 (TS < 750): {len(low_ts_df)}개 샘플")
    print(f"Mid TS 그룹 (750 <= TS < 1030): {len(mid_ts_df)}개 샘플")
    print(f"High TS 그룹 (TS >= 1030): {len(high_ts_df)}개 샘플")

    # Excel 파일로 저장
    print(f"\n분류된 데이터를 '{output_path}'에 저장 중...")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            low_ts_df.to_excel(writer, sheet_name='lowTS', index=False)
            print(f"  'lowTS' 시트에 {len(low_ts_df)}개 샘플 저장 완료.")
            
            mid_ts_df.to_excel(writer, sheet_name='midTS', index=False)
            print(f"  'midTS' 시트에 {len(mid_ts_df)}개 샘플 저장 완료.")

            high_ts_df.to_excel(writer, sheet_name='highTS', index=False)
            print(f"  'highTS' 시트에 {len(high_ts_df)}개 샘플 저장 완료.")
        
        print(f"\n✅ 분류 완료! 파일이 '{output_path}'에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")


def main():
    """
    메인 실행 함수
    """
    # 데이터 파일 경로 설정
    data_path = "TrainSet0507_NoDuplicatesHV.xlsx"
    output_path = "TrainSet0507_TSClassification.xlsx"
    
    classify_and_save_by_ts(data_path, output_path)


if __name__ == "__main__":
    main()
