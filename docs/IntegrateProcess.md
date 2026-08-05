```bash
SubmapAdvancedTSDF::IntegrateGPU(pts, nrm, cam)
│  CommandBatch batch(ctx)                       ← 배치 1개 생성
│
├─ m_base.IntegrateGPU(pts, nrm, cam, batch)     ← 모든 점 → base 맵
│     └─ [TiledAdvancedTSDF]
│          route(pts, nrm)  ─────────────────────← ★ 점을 타일로 분배 (+ghost)
│          for each 타일:
│             tileFor(key)  ─────────────────────← 타일 lazy 생성 (512³ 윈도우)
│             tile.RecordIntegrateGPU(sub, batch)← 배치에 dispatch "기록"만
│                └─ [AdvancedTSDF]
│                     ensureUploadCapacity(N) ───← 업로드 버퍼 여유있게 grow
│                     recordUpload(...) ─────────← 점/법선 memcpy + dispatch 기록
│
├─ denseSubset(pts, nrm) ────────────────────────← 밀집 블록 점만 추림
├─ m_detail.IntegrateGPU(densePts, batch)  ──────← 밀집 점 → detail 맵(half voxel), 동일 흐름
│
└─ batch.Submit()  ──────────────────────────────← ★ base+detail 통틀어 1번만 GPU 제출

```
