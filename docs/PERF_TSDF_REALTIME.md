# TSDF 실시간 성능 점검 (2026-08-01, Apple M4 Max / MoltenVK)

측정 도구: `example2/tsdf_benchmark`(합성 cube/cylinder, **동기 GPU 디스패치** wall-clock) + `tsdf_folder_eval`(실데이터 `scans/dragon`, 90프레임·248만 점). 모든 `SubmitOneShot`는 blocking이라 숫자는 보수적(파이프라이닝 여지 있음).

## 1. 핵심 수치 (합성, voxel 0.1, 8뷰·~6.1k점)

| 백엔드 | integrate | extract(=build−integ) | 메모리 | 비고 |
|---|---|---|---|---|
| Simple(weighted) | 6.1 ms | ~81 ms | 192 KB | 법선 없음(품질 최하) |
| **Directional(block)** | **33.0 ms** | ~38 ms | **6144 KB** | 4× 느림·18× 무거움 → 실시간 부적합 |
| Compact-Directional | 5.8 ms | ~53 ms | 329 KB | |
| **Advanced (우리 기본)** | **8.1 ms** | ~56 ms | **329 KB** | +A1/A2, 실시간 후보 |

→ **integrate 처리량 ≈ 1.3 µs/점 (~76만 점/초)**. block Directional만 4× 느림(블록 저장 오버헤드).

## 2. 실데이터 end-to-end (`scans/dragon`, voxel 0.5)
- 90프레임(프레임당 ~27.6k점, 총 248만 점) **적분 + 추출 1회 = 5.58 s** (RSS 533 MB).
- 분해(추정): ASCII PLY 읽기 ~1.5–2 s + 적분 ~3.2 s + 추출 ~0.5 s.
- **적분만 ≈ 36 ms / 밀집 27.6k점 프레임 → ~28 fps.**

## 3. 실시간 판정

| 시나리오 | 프레임 점수 | 적분 시간(추정) | fps | 판정 |
|---|---|---|---|---|
| 다운샘플 RGB-D | 5–15k | 7–20 ms | 50–140 | ✅ 여유 |
| 밀집 RGB-D/LiDAR | 27k | ~36 ms | ~28 | ✅ 실시간(경계) |
| 초밀집 | 100k+ | 130 ms+ | <8 | ⚠️ 다운샘플 필요 |

- **integrate는 스트리밍 실시간 가능** (전형적 프레임에서 25–140fps). 우리 **Advanced/Compact가 옳은 선택**, block Directional은 피할 것.
- **extract는 프레임마다 하면 안 됨**: 합성에서도 ~56ms, 실맵(290만 entry)에선 초 단위. **주기적/온디맨드**(뷰·메시)로만.

## 4. 병목 & 개선 여지 (측정 기반)
1. **동기 디스패치 오버헤드**: `SubmitOneShot`이 Integrate마다 blocking → 프레임당 고정비. **비동기/배치 제출**로 개선 가능(특히 소프레임 다수일 때).
2. **DownloadEntries가 디버그 병목**: `voxel_fill_debugger --dump`가 느린 건 알고리즘이 아니라 **매 프레임 290만 entry 다운로드+호스트 디프**(디버그 전용). 실파이프라인엔 없음.
3. **ASCII PLY 읽기**: 248만 점 텍스트 파싱이 오프라인 시간의 ~30% → **바이너리 PLY**로 대폭 단축.
4. **extract 전체 해시 스캔**: ROI/증분 추출로 실시간 뷰 가능(현재는 전량 스캔).
5. **UMA 이점**: M4 Max는 zero-copy라 대형 TSDF에 유리(디스크리트 GPU의 host↔device 복사 없음).

## 5. 논문 대비 위치
- **DB-TSDF**(02): CPU 상수 ~150 ms/scan. 우리는 GPU ~36 ms/밀집프레임 → **더 빠르되** 정밀도(연속거리·법선)는 우리가 우위.
- **ESLAM/PIN-SLAM**: 신경 SLAM은 GPU 실시간(수십 ms). 우리 명시적 TSDF는 integrate가 더 빠르고 결정적, 학습 불필요.
- **결론:** 볼류메트릭 실시간 매핑 클래스에서 **경쟁력 있음**. integrate는 실시간, extract만 주기화하면 30fps SLAM 매핑에 충분.

## 6. 실시간을 원할 때 체크리스트
- [ ] 백엔드 = **Advanced 또는 Compact-Directional** (block Directional 금지).
- [ ] 프레임 점수 다운샘플(voxel 다운샘플 또는 랜덤) → 30k↓ 권장.
- [ ] extract는 **매 N프레임** 또는 창 이동 시에만.
- [ ] I/O는 **바이너리 PLY**.
- [ ] 대용량은 `TiledAdvancedTSDF`(touched 타일만) — 단 per-tile hash 메모리 주의(별도 문서).
