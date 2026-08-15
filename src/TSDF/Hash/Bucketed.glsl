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

#define HASH_BUCKET_SIZE 32u

#define HASH_MAX_BUCKET_PROBE (MAX_PROBE / HASH_BUCKET_SIZE)

#ifdef HASH_WITH_INSERT
uint findOrInsert(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint bucketStep = 0u; bucketStep < HASH_MAX_BUCKET_PROBE; bucketStep++) {
		uint bucketBaseSlot = ((bucket + bucketStep) % bucketCount) * HASH_BUCKET_SIZE;
		for (uint slotInBucket = 0u; slotInBucket < HASH_BUCKET_SIZE; slotInBucket++) {
			uint index = bucketBaseSlot + slotInBucket;
			if (index >= g_hashCapacity) break;
			uint previousKey = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
			if (previousKey == EMPTY_KEY)
			{
				atomicAdd(g_filledCount, 1u);
				g_firstFrame[index] = g_currentFrame;
				countProbes(bucketStep * HASH_BUCKET_SIZE + slotInBucket + 1u);
				return index;
			}
			if (previousKey == key)
			{
				countProbes(bucketStep * HASH_BUCKET_SIZE + slotInBucket + 1u);
				return index;
			}
		}
	}
	countProbes(HASH_MAX_BUCKET_PROBE * HASH_BUCKET_SIZE);
	return HASH_INSERT_FAILED;
}
#endif

uint findSlot(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint bucketStep = 0u; bucketStep < HASH_MAX_BUCKET_PROBE; bucketStep++) {
		uint bucketBaseSlot = ((bucket + bucketStep) % bucketCount) * HASH_BUCKET_SIZE;
		bool sawEmpty = false;
		for (uint slotInBucket = 0u; slotInBucket < HASH_BUCKET_SIZE; slotInBucket++) {
			uint index = bucketBaseSlot + slotInBucket;
			if (index >= g_hashCapacity) break;
			uint found = g_hash[index].key;
			if (found == key) return index;
			if (found == EMPTY_KEY) sawEmpty = true;
		}
		if (sawEmpty) return HASH_NOT_FOUND;
	}
	return HASH_NOT_FOUND;
}
