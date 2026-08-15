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
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_hashCapacity;
		uint previousKey = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY)
		{
			atomicAdd(g_filledCount, 1u);
			// slot filled for the first time -> stamp the frame
			g_firstFrame[index] = g_currentFrame;
			countProbes(probeStep + 1u);
			return index;
		}
		if (previousKey == key)
		{
			countProbes(probeStep + 1u);
			return index;
		}
	}
	countProbes(MAX_PROBE);
	return HASH_INSERT_FAILED;
}

#endif

uint findSlot(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_hashCapacity;
		uint found = g_hash[index].key;
		if (found == EMPTY_KEY) return HASH_NOT_FOUND;
		if (found == key) return index;
	}
	return HASH_NOT_FOUND;
}
