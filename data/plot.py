import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# CSV 파일 경로 (사용자 환경에 맞게 수정)
csv_file = 'gaze_samples_20250831_050525_25x25_2.csv'

# CSV 파일 읽기
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"오류: '{csv_file}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# pt_index별로 그룹화
grouped = df.groupby('pt_index')

# 출력 디렉토리 생성 (이미지 저장용)
output_dir = 'gaze_plots_snapped'
os.makedirs(output_dir, exist_ok=True)

# 각 pt_index에 대해 두 플롯을 하나의 이미지에 나란히 저장
for pt_index, group in grouped:
    # Figure와 두 개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.3})

    # --- 첫 번째 서브플롯: uL, vL과 uR, vR ---
    ax1.scatter(group['uL'], group['vL'], alpha=0.5, color='blue', label='Left Eye')
    ax1.scatter(group['uR'], group['vR'], alpha=0.5, color='red', label='Right Eye')
    ax1.set_title(f'Recorded Eye Gaze: u vs v (Left: Blue, Right: Red) (pt_index={pt_index})')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal', adjustable='box')

    # --- 두 번째 서브플롯: Y_x, Y_y를 화면 위에 표시 ---
    if not group['screen_w'].empty:
        screen_w = group['screen_w'].iloc[0]
    else:
        screen_w = 2560 # 기본값
        
    if not group['screen_h'].empty:
        screen_h = group['screen_h'].iloc[0]
    else:
        screen_h = 1600 # 기본값

    # 격자 설정
    grid_divisions = 25
    x_ticks = np.arange(0, screen_w + 1, screen_w / grid_divisions)
    y_ticks = np.arange(0, screen_h + 1, screen_h / grid_divisions)

    # --- 요청사항: Gaze Point를 가장 가까운 격자 교차점으로 스냅 ---
    # 각 격자 칸의 너비와 높이 계산
    step_x = screen_w / grid_divisions
    step_y = screen_h / grid_divisions
    
    # Y_x, Y_y 좌표를 가장 가까운 교차점 좌표로 변환
    snapped_Y_x = (group['Y_x'] / step_x).round() * step_x
    snapped_Y_y = (group['Y_y'] / step_y).round() * step_y

    # 화면 크기의 네모 그리기
    ax2.add_patch(plt.Rectangle((0, 0), screen_w, screen_h, fill=False, edgecolor='black', linewidth=2))
    
    # 변환된 좌표에 큰 원으로 표시
    ax2.scatter(snapped_Y_x, snapped_Y_y, s=200, color='green', marker='o', label='Snapped Gaze Point')
    
    ax2.set_title(f'Calibration target Point on Screen (pt_index={pt_index})')
    ax2.set_xlabel('Y_x (Horizontal)')
    ax2.set_ylabel('Y_y (Vertical)')
    ax2.set_xlim(0, screen_w)
    ax2.set_ylim(screen_h, 0)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontsize=8)

    # 하나의 PNG 파일로 저장
    plt.savefig(os.path.join(output_dir, f'combined_gaze_pt_{pt_index}.png'), bbox_inches='tight')
    plt.close(fig)

print(f"이미지 저장이 완료되었습니다. '{output_dir}' 폴더를 확인해주세요.")