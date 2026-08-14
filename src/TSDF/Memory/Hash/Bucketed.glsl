/// *********************************************
/// Bucketed probing
///
/// The table is read as buckets of HASH_BUCKET_SIZE contiguous slots. A probe examines a whole
/// bucket before moving to the next, so a run touches far fewer cache lines than slot-at-a-time
/// linear probing at the same load -- which is what lets the growth threshold sit at 0.8.
///
/// Single-choice (no eviction, no second hash): an entry never moves once inserted, which the
/// integrate kernel depends on -- it accumulates into the slot with atomicAdd, so a relocation
/// would land those adds on another key's entry.
/// *********************************************

#define HASH_NAME "bucketed"
#define HASH_BUCKET_SIZE 32u

#ifdef HASH_WITH_INSERT
uint findOrInsert(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint b = 0u; b < MAX_PROBE; b++) {
		uint base = ((bucket + b) % bucketCount) * HASH_BUCKET_SIZE;
		for (uint j = 0u; j < HASH_BUCKET_SIZE; j++) {
			uint index = base + j;
			if (index >= g_hashCapacity) break;
			uint prev = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
			if (prev == EMPTY_KEY)
			{
				atomicAdd(g_filledCount, 1u);
				g_firstFrame[index] = g_currentFrame;
				return index;
			}
			if (prev == key) return index;
		}
	}
	return HASH_INSERT_FAILED;
}
#endif

uint findSlot(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint b = 0u; b < MAX_PROBE; b++) {
		uint base = ((bucket + b) % bucketCount) * HASH_BUCKET_SIZE;
		bool sawEmpty = false;
		for (uint j = 0u; j < HASH_BUCKET_SIZE; j++) {
			uint index = base + j;
			if (index >= g_hashCapacity) break;
			uint found = g_hash[index].key;
			if (found == key) return index;
			if (found == EMPTY_KEY) sawEmpty = true;
		}
		// An empty slot in this bucket means insertion would have stopped here, so the key
		// cannot live further along the probe run.
		if (sawEmpty) return HASH_NOT_FOUND;
	}
	return HASH_NOT_FOUND;
}
