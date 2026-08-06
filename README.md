# vkSpatial

- Vulkan 기반으로 3D 분석 알고리즘을 작성한다.

```bash
# 2. Release Mode로 빌드하기
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rel --target voxel_fill_debugger -j8
./build-rel/example2/voxel_fill_debugger --dir scan_out --voxel 0.5
```

## License

MIT License — see [LICENSE](LICENSE)
