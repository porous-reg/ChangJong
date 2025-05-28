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

# 사용자 정의: 표면 온도에 따른 총 입사 열유속 (W/m^2)
# Ts 온도가 높아질수록 들어오는 열량은 작아지는 Ae^(-m*Ts) 꼴 근사
# A = 20000 W/m^2, m = 0.002 1/K 로 가정하여 생성한 포인트
surface_temp_flux_points = np.array([300, 600, 900, 1200], dtype=float)  # K
heat_flux_values_at_points = np.array([10976, 6024, 3306, 1814], dtype=float) # W/m^2

def get_surface_heat_flux(T_surface_query):
    """표면 온도에 따른 총 입사 열유속 (q_surface_total) 선형 보간"""
    flux = np.interp(T_surface_query, surface_temp_flux_points, heat_flux_values_at_points, 
                     left=heat_flux_values_at_points[0], right=heat_flux_values_at_points[-1])
    return flux

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
frames = []
center_T = []
surface_mean_T = []
domain_mean_T = []
bottom_center_T = []
min_T_history = []
min_T_coords_history = []

# 초기 온도 (300K)에서의 물성치 참고값 계산
k_initial = get_k_at_temp(300.0)
cp_initial = get_cp_at_temp(300.0)
alpha_initial = k_initial / (rho * cp_initial)

# 물성치 및 주요 파라미터 출력 (h 관련 정보 제거)
print("--- Material Properties (Temperature-Dependent) ---")
print(f"Density (rho): {rho} kg/m³ (constant)")
print(f"Thermal conductivity (k): Varies with T. At 300K, k = {k_initial:.2f} W/(m·K)")
print(f"Specific heat capacity (cp): Varies with T. At 300K, cp = {cp_initial:.2f} J/(kg·K)")
print(f"Thermal diffusivity (alpha): Varies with T and position. At 300K, alpha = {alpha_initial:.2e} m²/s")
print("\n--- Grid & Simulation Parameters ---")
print(f"Grid size: Nx = {Nx}, Ny = {Ny}")
print(f"Grid spacing: dx = {dx:.4f} m, dy = {dy:.4f} m")
print(f"Time step (dt): {dt} s, Total time: {t_end} s, Number of steps: {Nt}")

print("\n--- Boundary Condition: Surface Heat Flux ---")
print(f"Heat flux is a function of surface temperature, interpolated from:")
for temp, flux_val in zip(surface_temp_flux_points, heat_flux_values_at_points):
    print(f"  T_surface = {temp:.0f} K -> Heat Flux = {flux_val:.0f} W/m²")

# 안정성 조건 확인 (CFL for 2D conduction) (초기 온도 300K 기준 참고값)
cfl_condition_initial = alpha_initial * dt * (1/dx**2 + 1/dy**2)
print(f"\nCFL-like stability criterion (alpha*dt*(1/dx^2 + 1/dy^2)): {cfl_condition_initial:.4f} (at 300K, should be <= 0.5 for explicit interior nodes)")
print("-----------------------------------\n")

# 시간 반복 계산
target_min_temp = 1260  # 목표 최소 온도 설정 (K)
reached_target_min_temp = False
time_at_target = 0
domain_mean_T_at_target = 0
surface_mean_T_at_target = 0
min_T_at_target = 0

total_heat_input = 0.0  # 총 투입 열량
total_heat_input_at_target = 0.0 # 목표 도달 시 총 투입 열량

for n in range(Nt):
    # 1. 내부 격자점 온도 계산 (Explicit FDM)
    for j in range(1, Ny):
        for i in range(1, Nx):
            T_current_node = T[j, i]
            k_node = get_k_at_temp(T_current_node)
            cp_node = get_cp_at_temp(T_current_node)
            alpha_node = k_node / (rho * cp_node)
            
            d2Tdx2 = (T[j, i + 1] - 2 * T[j, i] + T[j, i - 1]) / dx**2
            d2Tdy2 = (T[j + 1, i] - 2 * T[j, i] + T[j - 1, i]) / dy**2
            T_new[j, i] = T[j, i] + alpha_node * dt * (d2Tdx2 + d2Tdy2)

    # 2. 경계 조건 적용
    # 상단 벽 (y=Ly): 지정된 열유속
    j_top = Ny
    T_surface_top = T[j_top, :] 
    k_surface_top = get_k_at_temp(T_surface_top)
    q_total_top = get_surface_heat_flux(T_surface_top)
    T_new[j_top, :] = T[j_top - 1, :] + (q_total_top * dy) / k_surface_top

    # 좌측 벽 (x=0): 지정된 열유속
    i_left = 0
    T_surface_left = T[:, i_left]
    k_surface_left = get_k_at_temp(T_surface_left)
    q_total_left = get_surface_heat_flux(T_surface_left)
    T_new[:, i_left] = T[:, i_left + 1] + (q_total_left * dx) / k_surface_left
    
    # 우측 벽 (x=Lx): 지정된 열유속
    i_right = Nx
    T_surface_right = T[:, i_right]
    k_surface_right = get_k_at_temp(T_surface_right)
    q_total_right = get_surface_heat_flux(T_surface_right)
    T_new[:, i_right] = T[:, i_right - 1] + (q_total_right * dx) / k_surface_right
        
    # 하단 벽 (y=0): 단열 (가장 마지막에 적용하여 다른 조건에 의해 덮어쓰이지 않도록 함)
    T_new[0, :] = T_new[1, :]

    # --- 총 투입 열량 계산 (추가) ---
    # 각 표면에서 현재 시간 스텝(dt) 동안 들어온 열량 (J/m, 단위 깊이당)
    power_from_top = np.sum(q_total_top * dx)  # W/m
    power_from_left = np.sum(q_total_left * dy) # W/m
    power_from_right = np.sum(q_total_right * dy) # W/m
    
    current_step_power = power_from_top + power_from_left + power_from_right # Total power in W/m
    current_step_heat_input = current_step_power * dt # Total energy in J/m for this dt
    total_heat_input += current_step_heat_input
    # --- 계산 끝 ---

    # 업데이트된 온도로 T 배열 갱신
    T[:, :] = T_new[:, :]
    current_min_T = np.min(T) # 최소 온도는 업데이트된 T 기준

    # 목표 최소 온도 도달 확인
    if not reached_target_min_temp and current_min_T >= target_min_temp:
        reached_target_min_temp = True
        time_at_target = (n + 1) * dt # 현재 시간 스텝 완료 후 시간
        min_T_at_target = current_min_T
        
        surface_nodes_at_target = np.concatenate([
            T[Ny, :], T[1:Ny, 0], T[1:Ny, Nx]
        ])
        surface_mean_T_at_target = np.mean(surface_nodes_at_target)
        domain_mean_T_at_target = np.mean(T)
        total_heat_input_at_target = total_heat_input # 목표 도달 시점까지의 총 열량 저장
        
        print(f"\n--- Target Minimum Temperature Reached ---")
        print(f"Minimum temperature of {min_T_at_target:.2f} K reached at t = {time_at_target:.2f} s (Step {n+1}).")
        print(f"Domain Mean Temperature at target: {domain_mean_T_at_target:.2f} K")
        print(f"Surface Mean Temperature at target: {surface_mean_T_at_target:.2f} K")
        print(f"Total Heat Input at target: {total_heat_input_at_target:.2f} J/m (assuming unit depth)")
        
        # break # 필요시 시뮬레이션 중단 (주석 처리된 상태로 둠)
    
    # 지정된 프레임에서 데이터 저장
    if n in save_frames:
        frames.append(T.copy())
        center_T.append(T[Ny // 2, Nx // 2])
        
        # 표면 (상단, 좌측(하단 제외), 우측(하단 제외))의 노드 선택
        surface_nodes = np.concatenate([
            T[Ny, :],         # 상단 벽 전체
            T[1:Ny, 0],       # 좌측 벽 (y=0 제외, y=Ly는 상단벽에 포함)
            T[1:Ny, Nx]       # 우측 벽 (y=0 제외, y=Ly는 상단벽에 포함)
        ])
        surface_mean_T.append(np.mean(surface_nodes))
        domain_mean_T.append(np.mean(T))
        bottom_center_T.append(T[0, Nx // 2]) # 바닥면의 중앙 (y=0, x=Lx/2)
        min_T_history.append(np.min(T)) # 전체 도메인의 최소 온도
        min_idx = np.unravel_index(np.argmin(T), T.shape)
        min_T_coords_history.append((x[min_idx[1]], y[min_idx[0]]))

# GIF 애니메이션 저장
fig, ax = plt.subplots()
# vmin, vmax는 시뮬레이션 결과에 따라 조정하는 것이 좋습니다.
# 여기서는 초기 온도와 예상 가능한 최대 온도를 고려하여 설정합니다.
# 예를 들어, 300K (초기) 에서 열유입으로 온도가 상승할 것을 예상하여 1500K 정도로 설정
# 실제 최대 온도는 시뮬레이션 후 확인하고 조정할 수 있습니다.
v_min_anim = 300 
v_max_anim = 1500 # 예상 최대 온도로, 필요시 조정
if frames: # 프레임이 있는 경우에만
    v_min_anim = min(np.min(f) for f in frames) if frames else 300
    v_max_anim = max(np.max(f) for f in frames) if frames else 1500


cax = ax.imshow(frames[0] if frames else np.zeros((Ny+1, Nx+1)), cmap='hot', origin='lower', 
                extent=[0, Lx, 0, Ly], vmin=v_min_anim, vmax=v_max_anim)
fig.colorbar(cax, label='Temperature (K)')
ax.set_title("Temperature Distribution at t = 0 s (Specified Surface Heat Flux)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
if min_T_coords_history:
    min_T_marker, = ax.plot(min_T_coords_history[0][0], min_T_coords_history[0][1], 'rx', markersize=8, label='Min Temp Location')
else: # min_T_coords_history가 비어있을 경우 (save_frames가 비어있는 등) 대비
    min_T_marker, = ax.plot([], [], 'rx', markersize=8, label='Min Temp Location')


def update_heatmap(frame_idx):
    if not frames: return [cax] # 프레임이 없으면 업데이트 안함
    cax.set_array(frames[frame_idx])
    # cax.set_clim(vmin=np.min(frames[frame_idx]), vmax=np.max(frames[frame_idx])) # 프레임별로 min/max 동적 조정 시
    ax.set_title(f"Temperature Distribution at t = {int(save_frames[frame_idx]*dt)} s (Specified Surface Heat Flux)")
    if frame_idx < len(min_T_coords_history):
        min_x, min_y = min_T_coords_history[frame_idx]
        min_T_marker.set_data([min_x], [min_y])
    return [cax, min_T_marker]

if frames: # 프레임이 있을 때만 애니메이션 생성
    ani = animation.FuncAnimation(fig, update_heatmap, frames=len(frames), blit=True)
    ani.save("2D_plate_temperature_specified_flux_py.gif", writer="pillow", fps=20)
    print("GIF animation saved as 2D_plate_temperature_specified_flux_py.gif")
else:
    print("No frames to save for GIF animation.")

###########이거 맞나.
time_array = save_frames * dt
# 시간에 따른 주요 지점 온도 변화 플롯
if time_array.size > 0 : # 데이터가 있을 때만 플롯
    plt.figure(figsize=(12, 7))
    # plt.plot(time_array, center_T, label="Center Temperature (T_center)") # 필요시 주석 해제
    plt.plot(time_array, surface_mean_T, label="Surface Mean Temp. (T_surf_avg)")
    plt.plot(time_array, domain_mean_T, label="Domain Mean Temperature (T_domain_avg)")
    plt.plot(time_array, bottom_center_T, label="Bottom Center Temperature (T_bottom_center)")
    plt.plot(time_array, min_T_history, label="Minimum Temperature (T_min)")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Metrics Over Time (Specified Surface Heat Flux, Insulated Bottom)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temperature_metrics_plot_specified_flux_py.png")
    print("Temperature metrics plot saved as temperature_metrics_plot_specified_flux_py.png")
else:
    print("No data to plot for temperature metrics.")


# --- 온도 의존적 물성치 플롯 (Biot 수, h 관련 플롯 제거) ---
print("\nGenerating plots for temperature-dependent properties...")
plot_temps = np.linspace(temp_points[0], temp_points[-1], 200)
k_plot = get_k_at_temp(plot_temps)
cp_plot = get_cp_at_temp(plot_temps)
alpha_plot_vs_temp = k_plot / (rho * cp_plot) # 변수명 변경 (아래 CFL 플롯과 구분)

# 1. 열전도율(k) 및 비열(cp) vs. 온도
fig_props, ax1_props = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1_props.set_xlabel('Temperature (K)')
ax1_props.set_ylabel('Thermal Conductivity (k) [W/m·K]', color=color)
ax1_props.plot(plot_temps, k_plot, color=color, linestyle='-', label='k (Thermal Conductivity)')
ax1_props.tick_params(axis='y', labelcolor=color)
ax1_props.grid(True, linestyle='--', alpha=0.7)

ax2_props = ax1_props.twinx()
color = 'tab:blue'
ax2_props.set_ylabel('Specific Heat (cp) [J/kg·K]', color=color)
ax2_props.plot(plot_temps, cp_plot, color=color, linestyle='--', label='cp (Specific Heat)')
ax2_props.tick_params(axis='y', labelcolor=color)

fig_props.suptitle('Temperature-Dependent Material Properties', fontsize=14)
handles1, labels1 = ax1_props.get_legend_handles_labels()
handles2, labels2 = ax2_props.get_legend_handles_labels()
fig_props.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("material_properties_vs_temp_py.png")
print("Material properties plot saved as material_properties_vs_temp_py.png")

# 2. 열확산계수(alpha) vs. 온도
plt.figure(figsize=(10, 6))
plt.plot(plot_temps, alpha_plot_vs_temp, label=r'$\alpha = k / (\rho \cdot cp)$ (Thermal Diffusivity)', color='purple')
plt.xlabel("Temperature (K)")
plt.ylabel(r"Thermal Diffusivity ($\alpha$) [m²/s]")
plt.title("Thermal Diffusivity vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("thermal_diffusivity_vs_temp_py.png")
print("Thermal diffusivity plot saved as thermal_diffusivity_vs_temp_py.png")

# CFL 유사 안정성 조건 값 vs. 온도 (alpha가 변하므로 여전히 유용할 수 있음)
cfl_plot_vs_temp = alpha_plot_vs_temp * dt * (1/dx**2 + 1/dy**2) 
plt.figure(figsize=(10, 6))
plt.plot(plot_temps, cfl_plot_vs_temp, label=r'CFL-like value: $\alpha \cdot dt \cdot (1/dx^2 + 1/dy^2)$', color='green')
plt.axhline(y=0.5, color='r', linestyle='--', label='Stability Limit (0.5 for interior nodes)')
plt.xlabel("Temperature (K)")
plt.ylabel("CFL-like Value")
plt.title("CFL-like Stability Criterion vs. Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cfl_vs_temp_py.png")
print("CFL plot saved as cfl_vs_temp_py.png")

# 푸리에 수 플롯은 Lc 정의가 h에 의존했었으므로 제거됨.
print("Fourier number plot generation skipped as Lc definition was h-dependent or requires new Lc.")

simulation_actual_duration = (n + 1) * dt # 루프의 마지막 n 값을 사용

if reached_target_min_temp:
    print(f"\n--- Summary: Target Minimum Temperature ({target_min_temp} K) ---")
    print(f"First met or exceeded at t = {time_at_target:.2f} s.")
    print(f"Actual minimum temperature at that time: {min_T_at_target:.2f} K")
    print(f"Domain Mean Temperature at that time: {domain_mean_T_at_target:.2f} K")
    print(f"Surface Mean Temperature at that time: {surface_mean_T_at_target:.2f} K")
    print(f"Total Heat Input when target first met: {total_heat_input_at_target:.2f} J/m (assuming unit depth)")
elif t_end > 0 : # t_end가 0보다 큰 경우에만 "not reached" 메시지 표시
    print(f"\nTarget minimum temperature of {target_min_temp} K was not reached within the intended simulation time of {t_end:.2f} s.")
    print(f"Simulation completed. Final minimum temperature: {np.min(T):.2f} K at t = {simulation_actual_duration:.2f} s")


print(f"\n--- Overall Simulation Summary ---")
if t_end > 0:
    print(f"Simulation ran for a total duration of {simulation_actual_duration:.2f} s (out of intended {t_end:.2f} s).")
else: # t_end가 0일 경우 (예: Nt=0)
    print(f"Simulation ran for 0 steps (t_end = {t_end:.2f} s).")
    
print(f"Total accumulated heat input over this duration: {total_heat_input:.2f} J/m (assuming unit depth)")

# plt.show() # 이 줄을 주석 처리하거나 삭제합니다.
print("\nAll relevant plots generated and saved.") 