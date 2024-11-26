﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using FASTER.core;

namespace FASTER.stress
{
    interface IValueTester
    {
        long GetAverageSize();

        void PrepareTest();

        Task<bool> CheckpointStore();

        bool CompactStore();
    }

    internal interface IValueTester<TKey> : IValueTester, IDisposable
    {
        void Create(int hashTableCacheLines, LogSettings logSettings, CheckpointSettings checkpointSettings, IFasterEqualityComparer<TKey> comparer);

        void AddRecord(int keyOrdinal, ref TKey key);

        void TestRecord(int keyOrdinal, int keyCount, TKey[] keys);
        Task TestRecordAsync(int keyOrdinal, int keyCount, TKey[] keys);
    }
}
