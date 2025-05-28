import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 온도에 따른 물성치 데이터 (이미지 기반)
# T(K): [100, 200, 300, 400, 600, 800, 1000, 1200, 1500]
# C(kJ/kg K): [0.272, 0.402, 0.477, 0.505, 0.557, 0.582, 0.611, 0.640, 0.682]
# K(W/m K): [9.2, 12.6, 14.9, 16.6, 19.8, 22.6, 25.4, 28.0, 31.7]

temp_points = np.array([100, 200, 300, 400, 600, 800, 1000, 1200, 1500], dtype=float)
cp_points = np.array([0.272, 0.402, 0.477, 0.505, 0.557, 0.582, 0.611, 0.640, 0.682], dtype=float) * 1000  # J/kg·K 로 변환
k_points = np.array([9.2, 12.6, 14.9, 16.6, 19.8, 22.6, 25.4, 28.0, 31.7], dtype=float)

def get_k_at_temp(T_query):
    """온도에 따른 열전도율 (k) 선형 보간"""
    return np.interp(T_query, temp_points, k_points, left=k_points[0], right=k_points[-1])

def get_cp_at_temp(T_query):
    """온도에 따른 비열 (cp) 선형 보간"""
    return np.interp(T_query, temp_points, cp_points, left=cp_points[0], right=cp_points[-1])

# 밀도는 상수로 가정
rho = 8000  # 밀도 [kg/m³]

# 격자 및 시간 설정
Lx = 0.265 # x방향 길이 [m]
Ly = 0.365  # y방향 길이 [m]
Nx = 50  # x방향 격자점 수
Ny = 50  # y방향 격자점 수
dx = Lx / Nx  # x방향 격자 간격
dy = Ly / Ny  # y방향 격자 간격
x = np.linspace(0, Lx, Nx + 1)  # x 좌표
y = np.linspace(0, Ly, Ny + 1)  # y 좌표

dt = 0.1  # 시간 간격 [s]
t_end = 20000  # 총 시뮬레이션 시간 [s]
Nt = int(t_end / dt)  # 총 시간 스텝 수
save_frames = np.linspace(0, Nt - 1, 500, dtype=int)  # GIF 저장을 위한 프레임 선택

# 초기 온도 조건 (전체 300K)
T = np.ones((Ny + 1, Nx + 1)) * 300
T_new = T.copy()

# 결과 저장을 위한 리스트
frames = []  # 각 저장 시점의 온도 분포
center_T = []  # 중심점 온도 변화
surface_mean_T = []  # 대류 표면 평균 온도 변화
domain_mean_T = []  # 전체 도메인 평균 온도 변화
bottom_center_T = []  # 바닥면 중앙 온도 변화
min_T_history = []  # 전체 도메인 최소 온도 변화
min_T_coords_history = [] # 전체 도메인 최소 온도의 좌표 변화

# 경계 조건 파라미터
T_ref = 1300  # 외부 기준 온도 (대류) [K]
epsilon = 0.7  # 방사율 (주석 처리됨)
sigma = 5.67e-8  # 스테판-볼츠만 상수 (주석 처리됨)
h = 50.0  # 대류 열전달 계수 [W/(m²·K)]

# 초기 온도 (300K)에서의 물성치 참고값 계산
k_initial = get_k_at_temp(300.0)
cp_initial = get_cp_at_temp(300.0)
alpha_initial = k_initial / (rho * cp_initial)

# 물성치 및 주요 파라미터 출력
print("--- Material Properties (Temperature-Dependent) ---")
print(f"Density (rho): {rho} kg/m³ (constant)")
print(f"Thermal conductivity (k): Varies with T. At 300K, k = {k_initial:.2f} W/(m·K)")
print(f"Specific heat capacity (cp): Varies with T. At 300K, cp = {cp_initial:.2f} J/(kg·K)")
print(f"Thermal diffusivity (alpha): Varies with T and position. At 300K, alpha = {alpha_initial:.2e} m²/s")
print(f"Convective heat transfer coefficient (h): {h} W/(m²·K)")
print("\n--- Grid & Simulation Parameters ---")
print(f"Grid size: Nx = {Nx}, Ny = {Ny}")
print(f"Grid spacing: dx = {dx:.4f} m, dy = {dy:.4f} m")
print(f"Time step (dt): {dt} s, Total time: {t_end} s, Number of steps: {Nt}")

# 무차원 수 계산 및 출력 (초기 온도 300K 기준 참고값)
print("\n--- Dimensionless Numbers (Reference values at initial T=300K) ---")
# 특성 길이 (Lc) 계산: Lc = 고체 부피 / 대류 표면적
# 대류 표면: 상단(Lx), 좌측(Ly), 우측(Ly). 깊이는 단위 길이(1)로 가정.
# 하단은 단열이므로 대류 표면적 계산에서 제외됩니다.
convective_surface_area = Lx * 1 + Ly * 1 + Ly * 1 # Lx (상단) + Ly (좌측) + Ly (우측)
volume = Lx * Ly * 1 # 깊이 1로 가정
Lc = volume / convective_surface_area
print(f"Characteristic Length (Lc = V/As_conv): {Lc:.4f} m (As_conv = top + left + right sides)")

# 전체 Biot 수 (Bi_global): 고체 전체의 온도 분포 균일성 지표
# Bi_global = h * Lc / k
# Bi_global << 0.1 이면 고체 내부 온도가 비교적 균일하다고 간주 (집중용량계 해석 가능 근사 조건)
Bi_global_initial = (h * Lc) / k_initial
print(f"Global Biot Number (Bi_global = h*Lc/k): {Bi_global_initial:.4f} (at 300K) # Indicator of overall temperature uniformity")
print(f"  (A small Bi_global, e.g., << 0.1, suggests a relatively uniform temperature distribution within the solid)")

# 격자 기반 Biot 수 (Grid Biot Numbers): 수치 해석적 격자 수준 참고용 (초기 온도 300K 기준)
Bi_dx_grid_initial = (h * dx) / k_initial
Bi_dy_grid_initial = (h * dy) / k_initial
print(f"Grid Biot Number (dx based, Bi_dx_grid = h*dx/k): {Bi_dx_grid_initial:.4f} (at 300K) # For numerical/grid-level insight")
print(f"Grid Biot Number (dy based, Bi_dy_grid = h*dy/k): {Bi_dy_grid_initial:.4f} (at 300K) # For numerical/grid-level insight")

# 안정성 조건 확인 (CFL for 2D conduction) (초기 온도 300K 기준 참고값)
# 실제로는 경계조건의 영향도 고려해야 하며, alpha가 변하므로 국소적으로 CFL 조건이 달라질 수 있음.
cfl_condition_initial = alpha_initial * dt * (1/dx**2 + 1/dy**2)
print(f"CFL-like stability criterion (alpha*dt*(1/dx^2 + 1/dy^2)): {cfl_condition_initial:.4f} (at 300K, should be <= 0.5 for explicit interior nodes)")
print("-----------------------------------\n")

# 시간 반복 계산
for n in range(Nt):
    # 1. 내부 격자점 온도 계산 (Explicit FDM)
    # k, cp, alpha가 온도에 따라 변하므로, 각 격자점에서 국소적으로 계산 필요
    # T_new를 먼저 계산하고 T를 업데이트하기 때문에, 현재 시간스텝의 T를 기준으로 물성치를 계산.
    for j in range(1, Ny):
        for i in range(1, Nx):
            T_current_node = T[j, i]
            k_node = get_k_at_temp(T_current_node)
            cp_node = get_cp_at_temp(T_current_node)
            alpha_node = k_node / (rho * cp_node)
            
            # 참고: 아래 FDM 식은 alpha가 상수일 때 유도됨.
            # alpha가 공간적으로 변할 경우, rho*cp*dT/dt = div(k grad T)를 직접 이산화하는 것이 더 정확하나,
            # 여기서는 국소적 alpha를 사용하는 근사적 방법을 사용.
            d2Tdx2 = (T[j, i + 1] - 2 * T[j, i] + T[j, i - 1]) / dx**2
            d2Tdy2 = (T[j + 1, i] - 2 * T[j, i] + T[j - 1, i]) / dy**2
            T_new[j, i] = T[j, i] + alpha_node * dt * (d2Tdx2 + d2Tdy2)

    # 2. 경계 조건 적용
    # 상단 벽 (y=Ly): 대류
    j_top = Ny
    T_surface_top = T[j_top, :] # 현재 시간 스텝의 표면 온도
    k_surface_top = get_k_at_temp(T_surface_top)
    q_conv_top = h * (T_ref - T_surface_top)
    # q_rad_top = epsilon * sigma * (T_ref**4 - T_surface_top**4) # 복사 항 주석 처리
    q_total_top = q_conv_top
    # 경계조건 수식에서 k는 해당 표면에서의 k 값 사용
    T_new[j_top, :] = T[j_top - 1, :] + (q_total_top[:] * dy) / k_surface_top

    # 좌측 벽 (x=0): 대류
    i_left = 0
    for j_idx in range(Ny + 1): # y=0 부터 y=Ly 까지 모든 노드
        T_surface_left_node = T[j_idx, i_left] # 현재 시간 스텝의 표면 온도
        k_surface_left_node = get_k_at_temp(T_surface_left_node)
        q_conv_left = h * (T_ref - T_surface_left_node)
        # q_rad_left = epsilon * sigma * (T_ref**4 - T_surface_left_node**4) # 복사 항 주석 처리
        q_total_left = q_conv_left
        T_new[j_idx, i_left] = T[j_idx, i_left + 1] + (q_total_left * dx) / k_surface_left_node

    # 우측 벽 (x=Lx): 대류
    i_right = Nx
    for j_idx in range(Ny + 1): # y=0 부터 y=Ly 까지 모든 노드
        T_surface_right_node = T[j_idx, i_right] # 현재 시간 스텝의 표면 온도
        k_surface_right_node = get_k_at_temp(T_surface_right_node)
        q_conv_right = h * (T_ref - T_surface_right_node)
        # q_rad_right = epsilon * sigma * (T_ref**4 - T_surface_right_node**4) # 복사 항 주석 처리
        q_total_right = q_conv_right
        T_new[j_idx, i_right] = T[j_idx, i_right - 1] + (q_total_right * dx) / k_surface_right_node
        
    # 하단 벽 (y=0): 단열 (가장 마지막에 적용하여 다른 조건에 의해 덮어쓰이지 않도록 함)
    T_new[0, :] = T_new[1, :]

    # 업데이트된 온도로 T 배열 갱신
    T[:, :] = T_new[:, :]

    # 지정된 프레임에서 데이터 저장
    if n in save_frames:
        frames.append(T.copy())
        center_T.append(T[Ny // 2, Nx // 2])
        
        # 대류 표면 (상단, 좌측(하단 제외), 우측(하단 제외))의 노드 선택
        # 상단: T[Ny, :] (Ny+1 개 노드)
        # 좌측 (하단 모서리 제외): T[1:Ny, 0] (Ny-1 개 노드)
        # 우측 (하단 모서리 제외): T[1:Ny, Nx] (Ny-1 개 노드)
        # T[Ny,0]과 T[Ny,Nx]는 T[Ny,:]에 포함됨.
        convective_surface_nodes = np.concatenate([
            T[Ny, :],         # 상단 벽 전체
            T[1:Ny, 0],       # 좌측 벽 (y=0 제외, y=Ly는 상단벽에 포함)
            T[1:Ny, Nx]       # 우측 벽 (y=0 제외, y=Ly는 상단벽에 포함)
        ])
        surface_mean_T.append(np.mean(convective_surface_nodes))
        domain_mean_T.append(np.mean(T))
        bottom_center_T.append(T[0, Nx // 2]) # 바닥면의 중앙 (y=0, x=Lx/2)
        min_T_history.append(np.min(T)) # 전체 도메인의 최소 온도
        # 최소 온도의 좌표 찾기
        min_idx = np.unravel_index(np.argmin(T), T.shape)
        min_T_coords_history.append((x[min_idx[1]], y[min_idx[0]]))

# GIF 애니메이션 저장
fig, ax = plt.subplots()
# 초기 프레임 온도 범위를 기준으로 vmin, vmax 설정 또는 예상되는 최대 온도로 설정
# T_ref가 1300이므로, 최대 온도는 그 근처일 것으로 예상. 초기 300K.
cax = ax.imshow(frames[0], cmap='hot', origin='lower', extent=[0, Lx, 0, Ly], vmin=300, vmax=T_ref + 100)
fig.colorbar(cax, label='Temperature (K)')
ax.set_title("Temperature Distribution at t = 0 s")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# 초기 최소 온도 지점 표시
min_T_marker, = ax.plot(min_T_coords_history[0][0], min_T_coords_history[0][1], 'rx', markersize=8, label='Min Temp Location')

def update_heatmap(frame_idx):
    cax.set_array(frames[frame_idx])
    ax.set_title(f"Temperature Distribution at t = {int(save_frames[frame_idx]*dt)} s")
    # 최소 온도 지점 업데이트
    min_x, min_y = min_T_coords_history[frame_idx]
    min_T_marker.set_data([min_x], [min_y])
    return [cax, min_T_marker]

ani = animation.FuncAnimation(fig, update_heatmap, frames=len(frames), blit=True)
ani.save("2D_plate_temperature_convection_only_py.gif", writer="pillow", fps=20)
print("GIF animation saved as 2D_plate_temperature_convection_only_py.gif")

# 시간에 따른 주요 지점 온도 변화 플롯
time_array = save_frames * dt
plt.figure(figsize=(12, 7)) # 그래프 크기 조정
# plt.plot(time_array, center_T, label="Center Temperature (T_center)") # 중심점 온도 플롯 주석 처리
plt.plot(time_array, surface_mean_T, label="Convective Surface Mean Temp. (T_conv_surf_avg)")
plt.plot(time_array, domain_mean_T, label="Domain Mean Temperature (T_domain_avg)")
plt.plot(time_array, bottom_center_T, label="Bottom Center Temperature (T_bottom_center)")
plt.plot(time_array, min_T_history, label="Minimum Temperature (T_min)")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Metrics Over Time (Convection Walls, Insulated Bottom)")
plt.legend()
plt.grid(True)
plt.tight_layout() # 레이아웃 조정
plt.savefig("temperature_metrics_plot_convection_only_py.png")
print("Temperature metrics plot saved as temperature_metrics_plot_convection_only_py.png")

# --- 온도 의존적 물성치 및 무차원 수 플롯 추가 ---
print("\nGenerating plots for temperature-dependent properties and numbers...")
plot_temps = np.linspace(temp_points[0], temp_points[-1], 200) # 플롯을 위한 온도 범위

k_plot = get_k_at_temp(plot_temps)
cp_plot = get_cp_at_temp(plot_temps)
alpha_plot = k_plot / (rho * cp_plot)

# Lc는 스크립트 앞부분에서 이미 계산됨
Bi_global_plot = (h * Lc) / k_plot
Bi_dx_grid_plot = (h * dx) / k_plot
Bi_dy_grid_plot = (h * dy) / k_plot
cfl_plot = alpha_plot * dt * (1/dx**2 + 1/dy**2)

# 푸리에 수 계산을 위한 준비 (시뮬레이션 시간에 따라)
# time_array 와 domain_mean_T 는 메인 루프 후 이미 계산되어 있음.
k_mean_over_time = get_k_at_temp(np.array(domain_mean_T))
cp_mean_over_time = get_cp_at_temp(np.array(domain_mean_T))
alpha_mean_over_time = k_mean_over_time / (rho * cp_mean_over_time)

# Lc는 스크립트 앞부분에서 이미 계산된 전역 변수임
Fo_over_time = (alpha_mean_over_time * time_array) / (Lc**2)

# 1. 열전도율(k) 및 비열(cp) vs. 온도
fig_props, ax1_props = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1_props.set_xlabel('Temperature (K)')
ax1_props.set_ylabel('Thermal Conductivity (k) [W/(m·K)]', color=color)
ax1_props.plot(plot_temps, k_plot, color=color, linestyle='-', label='k (Thermal Conductivity)')
ax1_props.tick_params(axis='y', labelcolor=color)
ax1_props.grid(True, linestyle=':', which='both')

ax2_props = ax1_props.twinx()
color = 'tab:blue'
ax2_props.set_ylabel('Specific Heat Capacity (cp) [J/(kg·K)]', color=color)
ax2_props.plot(plot_temps, cp_plot, color=color, linestyle='--', label='cp (Specific Heat)')
ax2_props.tick_params(axis='y', labelcolor=color)

fig_props.suptitle('Temperature-Dependent Material Properties')
lines, labels = ax1_props.get_legend_handles_labels()
lines2, labels2 = ax2_props.get_legend_handles_labels()
ax2_props.legend(lines + lines2, labels + labels2, loc='center right')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("material_properties_vs_temp.png")
print("Plot of k and cp vs. temperature saved as material_properties_vs_temp.png")

# 2. 열확산율(alpha) vs. 온도
plt.figure(figsize=(10, 6))
plt.plot(plot_temps, alpha_plot, label='Alpha (Thermal Diffusivity)')
plt.xlabel("Temperature (K)")
plt.ylabel("Thermal Diffusivity (alpha) [m²/s]")
plt.title("Thermal Diffusivity vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("alpha_vs_temp.png")
print("Plot of alpha vs. temperature saved as alpha_vs_temp.png")

# 3. Biot 수 vs. 온도
plt.figure(figsize=(10, 6))
plt.plot(plot_temps, Bi_global_plot, label=f'Global Biot (Lc={Lc:.4f}m)')
plt.plot(plot_temps, Bi_dx_grid_plot, label=f'Grid Biot (dx={dx:.4f}m)', linestyle='--')
plt.plot(plot_temps, Bi_dy_grid_plot, label=f'Grid Biot (dy={dy:.4f}m)', linestyle=':')
plt.xlabel("Temperature (K)")
plt.ylabel("Biot Number (Bi = h*L/k)")
plt.title("Biot Numbers vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("biot_numbers_vs_temp.png")
print("Plot of Biot numbers vs. temperature saved as biot_numbers_vs_temp.png")

# 4. CFL 유사 안정성 조건 vs. 온도
plt.figure(figsize=(10, 6))
plt.plot(plot_temps, cfl_plot, label='CFL-like value')
plt.axhline(y=0.5, color='r', linestyle='--', label='Stability Limit (0.5)')
plt.xlabel("Temperature (K)")
plt.ylabel(f"CFL-like value (alpha*dt*(1/dx^2+1/dy^2)) for dt={dt}s")
plt.title("CFL-like Stability Criterion vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cfl_vs_temp.png")
print("Plot of CFL criterion vs. temperature saved as cfl_vs_temp.png")

# 5. 푸리에 수(Fo) vs. 시간
plt.figure(figsize=(10, 6))
plt.plot(time_array, Fo_over_time, label=f'Fourier Number (Lc={Lc:.4f}m)')
plt.xlabel("Time (s)")
plt.ylabel("Fourier Number (Fo = alpha * t / Lc^2)")
plt.title("Fourier Number vs. Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fourier_number_vs_time.png")
print("Plot of Fourier number vs. time saved as fourier_number_vs_time.png")

# --- 모든 플롯 표시 ---
plt.show()
