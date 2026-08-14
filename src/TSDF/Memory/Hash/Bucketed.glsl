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

// Probe budget, in BUCKETS. Deliberately derived from MAX_PROBE (the linear strategy's budget, in
// slots) divided by the bucket size, so both strategies give up after examining the SAME number of
// slots -- HASH_MAX_BUCKET_PROBE * HASH_BUCKET_SIZE == MAX_PROBE == 128 slots.
//
// Why parity rather than "MAX_PROBE buckets": the branch exists to compare the two strategies'
// memory and drop behaviour. Reusing MAX_PROBE as a bucket count would let bucketed examine
// 128 * 32 = 4096 slots against linear's 128, so at any table smaller than 4096 slots bucketed
// would sweep the whole table and structurally could not report an insert failure -- making
// "insertFailureCount == 0" mean something different for each strategy and turning the comparison
// into a probe-budget comparison instead of an addressing one. Equal slot budgets keep the drop
// counters directly comparable; what bucketed still buys over linear is cache locality within
// those 128 slots, which is what justifies its higher load-factor limit.
#define HASH_MAX_BUCKET_PROBE (MAX_PROBE / HASH_BUCKET_SIZE)

#ifdef HASH_WITH_INSERT
uint findOrInsert(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	// Budget: HASH_MAX_BUCKET_PROBE buckets x HASH_BUCKET_SIZE slots == MAX_PROBE slots examined,
	// the same slot budget linear probing gets. Do NOT compare this loop's trip count with
	// linear's without multiplying by HASH_BUCKET_SIZE first.
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
				return index;
			}
			if (previousKey == key) return index;
		}
	}
	return HASH_INSERT_FAILED;
}
#endif

uint findSlot(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	// Same budget as findOrInsert above: HASH_MAX_BUCKET_PROBE buckets == MAX_PROBE slots, matching
	// linear probing slot for slot. Lookup must never search a shorter run than insertion walked,
	// or a stored key becomes unreachable.
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
		// An empty slot in this bucket means insertion would have stopped here, so the key
		// cannot live further along the probe run.
		if (sawEmpty) return HASH_NOT_FOUND;
	}
	return HASH_NOT_FOUND;
}
