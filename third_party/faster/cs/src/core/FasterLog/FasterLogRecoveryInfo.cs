﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace FASTER.core
{
    /// <summary>
    /// Recovery info for FASTER Log
    /// </summary>
    public struct FasterLogRecoveryInfo
    {
        /// <summary>
        /// FasterLog recovery version
        /// </summary>
        const int FasterLogRecoveryVersion = 1;
        
        /// <summary>
        /// Begin address
        /// </summary>
        public long BeginAddress;

        /// <summary>
        /// Flushed logical address
        /// </summary>
        public long UntilAddress;

        /// <summary>
        /// Persisted iterators
        /// </summary>
        public Dictionary<string, long> Iterators;

        /// <summary>
        /// User-specified commit cookie
        /// </summary>
        public byte[] Cookie;
        
        /// <summary>
        /// commit num unique to this commit request
        /// </summary>
        public long CommitNum;
        
        /// <summary>
        /// whether this request is from a strong commit
        /// </summary>
        public bool FastForwardAllowed;

        /// <summary>
        /// callback to invoke when commit is presistent
        /// </summary>
        public Action Callback;
        
        /// <summary>
        /// Initialize
        /// </summary>
        public void Initialize()
        {
            BeginAddress = 0;
            UntilAddress = 0;
            Iterators = null;
            Cookie = null;
        }
        

        /// <summary>
        /// Initialize from stream
        /// </summary>
        /// <param name="reader"></param>
        public void Initialize(BinaryReader reader)
        {
            int version;
            long checkSum;
            try
            {
                version = reader.ReadInt32();
                checkSum = reader.ReadInt64();
                BeginAddress = reader.ReadInt64();
                UntilAddress = reader.ReadInt64();
                if (version > 0)
                    CommitNum = reader.ReadInt64();
                else
                    CommitNum = -1;
            }
            catch (Exception e)
            {
                throw new FasterException("Unable to recover from previous commit. Inner exception: " + e.ToString());
            }
            if (version < 0 || version > FasterLogRecoveryVersion)
                throw new FasterException("Invalid version found during commit recovery");

            var iteratorCount = 0;
            try
            {
                iteratorCount = reader.ReadInt32();
            }
            catch { }

            if (iteratorCount > 0)
            {
                Iterators = new Dictionary<string, long>();
                for (int i = 0; i < iteratorCount; i++)
                {
                    Iterators.Add(reader.ReadString(), reader.ReadInt64());
                }
            }

            int cookieLength = -1;
            long cookieChecksum = 0;
            if (version >= FasterLogRecoveryVersion)
            {
                try
                {
                    cookieLength = reader.ReadInt32();
                }
                catch { }

                if (cookieLength >= 0)
                {
                    Cookie = reader.ReadBytes(cookieLength);
                    unsafe
                    {
                        fixed (byte* ptr = Cookie)
                            cookieChecksum = (long)Utility.XorBytes(ptr, cookieLength);
                    }
                }
            }

            long computedChecksum = BeginAddress ^ UntilAddress;
            if (version >= FasterLogRecoveryVersion)
                computedChecksum ^= CommitNum ^ iteratorCount ^ cookieLength ^ cookieChecksum;

            // Handle case where all fields are zero
            if (version == 0 && BeginAddress == 0 && UntilAddress == 0 && iteratorCount == 0)
                throw new FasterException("Invalid checksum found during commit recovery");

            if (checkSum != computedChecksum)
                throw new FasterException("Invalid checksum found during commit recovery");
        }

        /// <summary>
        /// Reset
        /// </summary>
        public void Reset()
        {
            Initialize();
        }

        /// <summary>
        /// Write info to byte array
        /// </summary>
        public readonly byte[] ToByteArray()
        {
            using MemoryStream ms = new();
            using (BinaryWriter writer = new(ms))
            {
                writer.Write(FasterLogRecoveryVersion); // version

                int iteratorCount = 0;
                if (Iterators != null) iteratorCount = Iterators.Count;

                int cookieLength = -1;
                long cookieChecksum = 0;
                if (Cookie != null)
                {
                    cookieLength = Cookie.Length;
                    if (cookieLength > 0)
                        unsafe
                        {
                            fixed (byte *ptr = Cookie)
                                cookieChecksum = (long)Utility.XorBytes(ptr, cookieLength);
                        }
                }

                writer.Write(BeginAddress ^ UntilAddress ^ CommitNum ^ iteratorCount ^ cookieLength ^ cookieChecksum); // checksum
                writer.Write(BeginAddress);
                writer.Write(UntilAddress);
                writer.Write(CommitNum);

                writer.Write(iteratorCount);
                if (iteratorCount > 0)
                {
                    foreach (var kvp in Iterators)
                    {
                        writer.Write(kvp.Key);
                        writer.Write(kvp.Value);
                    }
                }

                writer.Write(cookieLength);
                if (cookieLength > 0)
                    writer.Write(Cookie);
            }
            return ms.ToArray();
        }

        /// <summary>
        /// </summary>
        /// <returns> size of this recovery info serialized </returns>
        public int SerializedSize()
        {
            var iteratorSize = sizeof(int);
            if (Iterators != null)
            {
                foreach (var kvp in Iterators)
                    iteratorSize += kvp.Key.Length + sizeof(long);
            }

            return sizeof(int) + 4 * sizeof(long) + iteratorSize + sizeof(int) + (Cookie?.Length ?? 0);
        }

        /// <summary>
        /// Take snapshot of persisted iterators
        /// </summary>
        /// <param name="persistedIterators">Persisted iterators</param>
        public void SnapshotIterators(ConcurrentDictionary<string, FasterLogScanIterator> persistedIterators)
        {
            Iterators = new Dictionary<string, long>();

            if (persistedIterators.Count > 0)
            {
                foreach (var kvp in persistedIterators)
                {
                    Iterators.Add(kvp.Key, kvp.Value.requestedCompletedUntilAddress);
                }
            }
        }

        /// <summary>
        /// Update iterators after persistence
        /// </summary>
        /// <param name="persistedIterators">Persisted iterators</param>
        public void CommitIterators(ConcurrentDictionary<string, FasterLogScanIterator> persistedIterators)
        {
            if (Iterators?.Count > 0)
            {
                foreach (var kvp in Iterators)
                {
                    persistedIterators[kvp.Key].UpdateCompletedUntilAddress(kvp.Value);
                }
            }
        }

        /// <summary>
        /// Print checkpoint info for debugging purposes
        /// </summary>
        public void DebugPrint()
        {
            Debug.WriteLine("******** Log Commit Info ********");

            Debug.WriteLine("BeginAddress: {0}", BeginAddress);
            Debug.WriteLine("FlushedUntilAddress: {0}", UntilAddress);
        }
    }
}
