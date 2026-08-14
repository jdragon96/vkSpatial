#version 460

/*
희소한 HashTable
[빈칸][voxel][빈칸][voxel][voxel][빈칸]...
                  ↓ Compact
조밀한 출력 배열
[voxel][voxel][voxel]

- 이산적으로 떨어져있는 Voxel buffer를 linear하게 이어주는 역할을 한다.
*/

/// *********************************************
/// Constants
/// *********************************************
layout(local_size_x = 64) in;

#define TSDF_SCALE  10000
#define MIN_WEIGHT  (TSDF_SCALE / 2)

#include "voxel_common.glsl"   // EMPTY_KEY

// 24 bytes. Layout must match DirEntry in AdvancedTSDF.integrate.comp.glsl and AdvancedTSDF.h.
struct DirEntry
{
	uint key;
	int  sumDW;   // Σ tsdf · w · TSDF_SCALE
	uint sumW;    // Σ w · TSDF_SCALE
	int  sumNx;   // Σ n · w · TSDF_SCALE  (stored gradient)
	int  sumNy;
	int  sumNz;
};

// 36 bytes. Layout must match AdvancedEntry in AdvancedTSDF.h EXACTLY (the CPU memcpy's the readback
// straight into std::vector<AdvancedEntry> -- no per-entry CPU decode). Scalars only, so std430 packs
// them tightly (a vec3 would force 16-byte alignment and break the match).
struct OutEntry
{
	float cx, cy, cz;   // world-space voxel centre
	uint  dir;          // dominant direction (key & 0x7)
	float tsdf;         // sumDW / sumW
	float weight;       // sumW / TSDF_SCALE
	float nx, ny, nz;   // normalized stored gradient (0 if degenerate)
	int   firstFrame;   // frame that first filled this slot (from g_firstFrame)
};

layout(std430, set = 0, binding = 0) readonly buffer HashTable  { DirEntry g_hash[];      };
layout(std430, set = 0, binding = 1)          buffer Compact    { OutEntry g_out[];       };
layout(        set = 0, binding = 2)          buffer Counter    { uint     g_count;       };
layout(std430, set = 0, binding = 3) readonly buffer FirstFrame { int      g_firstFrame[]; };

layout(push_constant) uniform PC {
	uint  g_hashCapacity;
	uint  g_maxOut;       // capacity of g_out in entries; append stops here (overflow is detected via g_count)
	float g_voxelSize;
	int   g_originX, g_originY, g_originZ;   // this tile's origin voxel (local key coords -> world voxel)
	int   g_coreMinX, g_coreMinY, g_coreMinZ; // append only local voxels in [coreMin, coreMax) --
	int   g_coreMaxX, g_coreMaxY, g_coreMaxZ; // moves the tiled ghost-dedup into the kernel
};

void main()
{
	uint slot = gl_GlobalInvocationID.x;
	if (slot >= g_hashCapacity) return;

	DirEntry entry = g_hash[slot];
	if (entry.key == EMPTY_KEY)        return;   // 1. empty slot
	if (entry.sumW < uint(MIN_WEIGHT)) return;   // 2. too few observations to be a surface

	// 3. unpack the local voxel + direction from the key (mirrors DecodeEntry / the integrate packing).
	int dir = int(entry.key & 0x7u);
	int localKeyZ  = int((entry.key >>  3u) & 0x1FFu);
	int localKeyY  = int((entry.key >> 12u) & 0x1FFu);
	int localKeyX  = int((entry.key >> 21u) & 0x1FFu);

	// 4. core-only: skip voxels outside this tile's owned region (a neighbour owns them in its core).
	if (localKeyX < g_coreMinX || localKeyX >= g_coreMaxX) return;
	if (localKeyY < g_coreMinY || localKeyY >= g_coreMaxY) return;
	if (localKeyZ < g_coreMinZ || localKeyZ >= g_coreMaxZ) return;

	// 5. reserve an output slot. g_count counts EVERY core hit (even past g_maxOut), so the CPU reads
	//    it back as the exact required size and can grow + redo if the shared buffer was too small.
	uint outIndex = atomicAdd(g_count, 1u);
	if (outIndex >= g_maxOut) return;

	// 6. decode to world space and write the ready-made AdvancedEntry.
	vec3 centre = vec3(
		float(localKeyX + g_originX), 
		float(localKeyY + g_originY), 
		float(localKeyZ + g_originZ));
	centre += vec3(0.5);
	centre *= g_voxelSize;
	float w      = float(entry.sumW);
	vec3  n      = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
	float nlen   = length(n);
	vec3  normal = nlen > 1e-6 ? n / nlen : vec3(0.0);

	g_out[outIndex].cx     = centre.x;
	g_out[outIndex].cy     = centre.y;
	g_out[outIndex].cz     = centre.z;
	g_out[outIndex].dir    = uint(dir);
	g_out[outIndex].tsdf   = float(entry.sumDW) / w;
	g_out[outIndex].weight = w / float(TSDF_SCALE);
	g_out[outIndex].nx     = normal.x;
	g_out[outIndex].ny     = normal.y;
	g_out[outIndex].nz     = normal.z;
	g_out[outIndex].firstFrame = g_firstFrame[slot];
}
