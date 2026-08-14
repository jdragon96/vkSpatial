/// *********************************************
/// Linear probing (current behaviour)
///
/// One slot per probe from wangHash(key), walking forward. Cheap at low load, but the probe
/// run grows quadratically as the table fills -- which is why its growth threshold is 0.5.
/// *********************************************

#define HASH_NAME "linear"

#ifdef HASH_WITH_INSERT
uint findOrInsert(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint p = 0u; p < MAX_PROBE; p++) {
		uint index = (slot + p) % g_hashCapacity;
		uint prev = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
		if (prev == EMPTY_KEY)
		{
			atomicAdd(g_filledCount, 1u);
			g_firstFrame[index] = g_currentFrame; // slot filled for the first time -> stamp the frame
			return index;
		}
		if (prev == key) return index;
	}
	return HASH_INSERT_FAILED;
}
#endif

uint findSlot(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint p = 0u; p < MAX_PROBE; p++) {
		uint index = (slot + p) % g_hashCapacity;
		uint found = g_hash[index].key;
		if (found == EMPTY_KEY) return HASH_NOT_FOUND;
		if (found == key) return index;
	}
	return HASH_NOT_FOUND;
}
