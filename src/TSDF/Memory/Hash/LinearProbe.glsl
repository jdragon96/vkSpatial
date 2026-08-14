/// *********************************************
/// Linear probing (current behaviour)
///
/// One slot per probe from wangHash(key), walking forward. Cheap at low load, but the probe
/// run grows quadratically as the table fills -- which is why its growth threshold is 0.5.
/// *********************************************

#ifdef HASH_WITH_INSERT
uint findOrInsert(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	// Budget: MAX_PROBE (128) SLOTS. Bucketed.glsl states its budget in the same unit so the two
	// strategies' drop counters stay directly comparable.
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_hashCapacity;
		uint previousKey = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY)
		{
			atomicAdd(g_filledCount, 1u);
			g_firstFrame[index] = g_currentFrame; // slot filled for the first time -> stamp the frame
			return index;
		}
		if (previousKey == key) return index;
	}
	return HASH_INSERT_FAILED;
}
#endif

uint findSlot(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	// Same budget as findOrInsert above: MAX_PROBE (128) SLOTS. See Bucketed.glsl for why the two
	// strategies are held to the same slot budget rather than the same loop trip count.
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_hashCapacity;
		uint found = g_hash[index].key;
		if (found == EMPTY_KEY) return HASH_NOT_FOUND;
		if (found == key) return index;
	}
	return HASH_NOT_FOUND;
}
