# 02. DB-TSDF — 거리를 "비트(0/1)"로 저장해 CPU만으로 빠르게 (심화)
**논문:** DB-TSDF: Directional Bitmask-based Truncated Signed Distance Fields (Maese, Merino, Caballero, 2025) · arXiv:2509.20081
**PDF:** `../02_DB-TSDF_DirectionalBitmask_2025.pdf`

## 🧒 한 줄 요약
거리를 실수로 계산하지 말고 **32칸 스위치**로: 켜진 스위치 개수 = 거리. 갱신은 **AND 한 번**(곱셈·나눗셈 없음) → GPU 없이 상수시간.

---

## 0. 문제 설정
실수 TSDF는 매 복셀마다 (거리, 가중치) 실수 연산 + 부동소수 저장이 필요해 CPU 고해상도에서 느림. DB-TSDF는 **정수 비트연산만** 쓰고, **스캔당 비용이 격자 크기와 무관**하도록 설계.

## 1. 복셀 레이아웃 (8바이트)
$$M_{\text{grid}}=N_x N_y N_z\cdot S_{\text{voxel}},\qquad S_{\text{voxel}}=8\text{ B}=\underbrace{32\text{b}}_{\text{거리마스크}}+\underbrace{1\text{b}}_{\text{부호}}+\underbrace{8\text{b}}_{\text{히트카운터}}(+\text{패딩}).$$

## 2. 비트마스크 거리 인코딩 (핵심)

### 2.1 정의
커널 중심에서 격자거리 $r=\sqrt{x^2+y^2+z^2}$ (voxel 단위)인 셀의 마스크:
$$m(x,y,z)=\begin{cases}0,& r=0\\[4pt](2^{32}-1)\gg\big(32-\lceil r\rceil\big),& r>0.\end{cases}$$
$2^{32}-1$ 은 32비트 전부 1. 오른쪽 시프트 $\gg(32-\lceil r\rceil)$ 하면 **위쪽 $\lceil r\rceil$ 개 비트만 1**이 남음.

### 2.2 "켜진 비트 수 = 거리"의 증명
$m$ 의 popcount(1의 개수):
$$\operatorname{popcount}\!\big((2^{32}-1)\gg(32-b)\big)=b,\quad b=\lceil r\rceil.$$
따라서 마스크에서 거리를 읽으려면 **popcount 한 번**: $\hat r=\operatorname{popcount}(m)$. 거리를 부동소수로 저장·계산할 필요가 없음.

> 🧒 자를 눈금 숫자가 아니라 **켜진 전구 개수**로 읽어요. 전구는 껐다 켜기만 하니 계산이 안 필요해요.

## 3. 갱신 = AND (거리는 단조 감소)

새 관측(커널) 마스크 $m_{\text{kernel}}$ 과 저장값을 AND:
$$m_{\text{grid}}\leftarrow \begin{cases}m_{\text{grid}}\,\&\,m_{\text{kernel}},& m_{\text{grid}}\ne m_{\text{grid}}\,\&\,m_{\text{kernel}}\\ m_{\text{grid}},&\text{else (쓰기 생략).}\end{cases}$$

### 3.1 왜 항상 "가장 가까운 거리"가 남나 (증명)
AND는 비트를 **끄기만** 함: $\forall b,\ (a\,\&\,b)\le a$ (비트 단조성). popcount도 단조: $\operatorname{popcount}(m\,\&\,m')\le\operatorname{popcount}(m)$. 즉
$$\hat r_{\text{new}}=\operatorname{popcount}(m\,\&\,m_{\text{kernel}})\le \hat r_{\text{old}}.$$
새 커널이 더 가까운 표면을 증거하면 거리가 줄고, 아니면 그대로. → 여러 스캔의 **min 거리**를 자동 유지(별도 min 연산 불필요). 쓰기 생략 조건으로 캐시/메모리 트래픽도 절감.

## 4. 방향 커널 + 그림자 (부호)

### 4.1 방향 양자화
센서프레임 점 $\mathbf p=(x,y,z)$ 의 방위·고도 빈:
$$b_{\text{az}}=\Big\lfloor\tfrac{\operatorname{atan2}(y,x)}{2\pi}B_{\text{az}}\Big\rfloor,\quad b_{\text{el}}=\Big\lfloor\tfrac{\arcsin(z/\lVert\mathbf p\rVert)+\pi/2}{\pi}B_{\text{el}}\Big\rfloor.$$
$B_{\text{az}}=B_{\text{el}}=40\Rightarrow1600$ 빈, 각 빈마다 $21^3$ 커널을 **미리 계산**해 둠(런타임 기하연산 0).

### 4.2 점유(부호): 반구 그림자 + 포화 카운터
표면 뒤쪽(자유공간이 아닌 곳)에서만 히트 카운터 증가:
$$h^{\text{new}}=\min(h^{\text{old}}+\mathbf 1_{\text{shadow}},\ H_{\max}),\qquad s^{\text{new}}=\begin{cases}s^{\text{new}}=\text{occupied},& h^{\text{old}}<T\wedge h^{\text{new}}\ge T\\ s^{\text{old}},&\text{else.}\end{cases}$$
$T$ 회 이상 관측돼야 "점유" 확정 → 잡음 한 번에 안 흔들림. 반구(hemisphere) 그림자는 접촉 셀 주변 정면 반구만 채워, 적은 프레임에서도 표면 연속성 확보(원뿔보다 빠른 수렴).

## 5. 상수 시간 비용
$$T_{\text{update}}\approx N_p\cdot K^3\cdot C_{\text{op}}.$$
$K=21$ 고정, $C_{\text{op}}$=AND+대입. **격자 해상도 $N_x N_y N_z$ 가 식에 없음** → voxel 0.3→0.05 m로 바꿔도 시간 ≈일정(~150 ms/scan), 메모리만 증가. 실측 Fig.7이 이 불변성을 보임.

> 🧒 지도를 아무리 촘촘히 해도 "점 하나당 21×21×21 스위치 끄기"만 하니 **시간이 안 늘어요**.

## 6. 우리 프로젝트와의 관계
- "방향별 저장"은 우리 Directional/Advanced TSDF와 같은 철학, 그러나 **실수 가중평균(01번 §0.2)** 대신 **비트마스크+AND(min)** 로 대체 → 정밀도(연속 거리) 일부를 내주고 CPU 상수시간·8B/복셀을 얻음.
- 우리 24B/복셀(그래디언트 포함) 대비 8B로 초경량. 대신 법선 품질·서브복셀 정밀도는 우리 stored-gradient가 유리.
