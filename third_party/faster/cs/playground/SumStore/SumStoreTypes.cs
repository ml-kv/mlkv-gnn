﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Threading;
using FASTER.core;

namespace SumStore
{
    public struct AdId : IFasterEqualityComparer<AdId>
    {
        public long adId;

        public long GetHashCode64(ref AdId key)
        {
            return Utility.GetHashCode(key.adId);
        }
        public bool Equals(ref AdId k1, ref AdId k2)
        {
            return k1.adId == k2.adId;
        }
    }

    public struct Input
    {
        public AdId adId;
        public NumClicks numClicks;
    }

    public struct NumClicks
    {
        public long numClicks;
    }

    public struct Output
    {
        public NumClicks value;
    }

    public sealed class Functions : FunctionsBase<AdId, NumClicks, Input, Output, Empty>
    {
        public override void CheckpointCompletionCallback(int sessionID, string sessionName, CommitPoint commitPoint)
        {
            Console.WriteLine($"Session {sessionID} ({(sessionName ?? "null")}) reports persistence until {commitPoint.UntilSerialNo}");
        }

        // Read functions
        public override bool SingleReader(ref AdId key, ref Input input, ref NumClicks value, ref Output dst, ref ReadInfo readInfo)
        {
            dst.value = value;
            return true;
        }

        public override bool ConcurrentReader(ref AdId key, ref Input input, ref NumClicks value, ref Output dst, ref ReadInfo readInfo)
        {
            dst.value = value;
            return true;
        }

        // RMW functions
        public override bool InitialUpdater(ref AdId key, ref Input input, ref NumClicks value, ref Output output, ref RMWInfo rmwInfo)
        {
            value = input.numClicks;
            return true;
         }

        public override bool InPlaceUpdater(ref AdId key, ref Input input, ref NumClicks value, ref Output output, ref RMWInfo rmwInfo)
        {
            Interlocked.Add(ref value.numClicks, input.numClicks.numClicks);
            return true;
        }

        public override bool CopyUpdater(ref AdId key, ref Input input, ref NumClicks oldValue, ref NumClicks newValue, ref Output output, ref RMWInfo rmwInfo)
        {
            newValue.numClicks = oldValue.numClicks + input.numClicks.numClicks;
            return true;        }
    }
}
