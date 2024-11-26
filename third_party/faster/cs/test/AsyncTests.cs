﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Threading;
using System.Threading.Tasks;
using FASTER.core;
using System.IO;
using NUnit.Framework;
using FASTER.test.recovery.sumstore;

namespace FASTER.test.async
{
    [TestFixture]
    public class AsyncRecoveryTests
    {
        private FasterKV<AdId, NumClicks> fht1;
        private FasterKV<AdId, NumClicks> fht2;
        private readonly AdSimpleFunctions functions = new AdSimpleFunctions();
        private IDevice log;


        [TestCase(CheckpointType.FoldOver)]
        [TestCase(CheckpointType.Snapshot)]
        [Category("FasterKV"), Category("CheckpointRestore")]
        [Category("Smoke")]

        public async Task AsyncRecoveryTest1(CheckpointType checkpointType)
        {
            TestUtils.DeleteDirectory(TestUtils.MethodTestDir, wait:true);
            log = Devices.CreateLogDevice(TestUtils.MethodTestDir + "/AsyncRecoveryTest1.log", deleteOnClose: true);

            string testPath = TestUtils.MethodTestDir + "/checkpoints4";
            Directory.CreateDirectory(testPath);

            fht1 = new FasterKV<AdId, NumClicks>
                (128,
                logSettings: new LogSettings { LogDevice = log, MutableFraction = 0.1, PageSizeBits = 10, MemorySizeBits = 13 },
                checkpointSettings: new CheckpointSettings { CheckpointDir = testPath }
                );

            fht2 = new FasterKV<AdId, NumClicks>
                (128,
                logSettings: new LogSettings { LogDevice = log, MutableFraction = 0.1, PageSizeBits = 10, MemorySizeBits = 13 },
                checkpointSettings: new CheckpointSettings { CheckpointDir = testPath }
                );

            int numOps = 5000;
            var inputArray = new AdId[numOps];
            for (int i = 0; i < numOps; i++)
            {
                inputArray[i].adId = i;
            }

            NumClicks value;
            AdInput inputArg = default;
            Output output = default;

            var s0 = fht1.For(functions).NewSession<AdSimpleFunctions>();
            var s1 = fht1.For(functions).NewSession<AdSimpleFunctions>();
            var s2 = fht1.For(functions).NewSession<AdSimpleFunctions>();

            for (int key = 0; key < numOps; key++)
            {
                value.numClicks = key;
                s1.Upsert(ref inputArray[key], ref value, Empty.Default, key);
            }

            for (int key = 0; key < numOps; key++)
            {
                value.numClicks = key;
                s2.Read(ref inputArray[key], ref inputArg, ref output, Empty.Default, key);
            }

            // does not require session
            fht1.TryInitiateFullCheckpoint(out _, checkpointType);
            await fht1.CompleteCheckpointAsync();

            s2.CompletePending(true,false);

            fht1.TryInitiateFullCheckpoint(out Guid token, checkpointType);
            await fht1.CompleteCheckpointAsync();

            s2.Dispose();
            s1.Dispose();
            s0.Dispose();
            fht1.Dispose();

            fht2.Recover(token); // sync, does not require session

            using (var s3 = fht2.For(functions).ResumeSession<AdSimpleFunctions>(s1.ID, out CommitPoint lsn))
            {
                Assert.AreEqual(numOps - 1, lsn.UntilSerialNo);

                for (int key = 0; key < numOps; key++)
                {
                    var status = s3.Read(ref inputArray[key], ref inputArg, ref output, Empty.Default, s3.SerialNo);

                    if (status.IsPending)
                        s3.CompletePending(true,true);
                    else
                    {
                        Assert.AreEqual(key, output.value.numClicks);
                    }
                }
            }

            fht2.Dispose();
            log.Dispose();
            TestUtils.DeleteDirectory(TestUtils.MethodTestDir);
        }
    }

    public class AdSimpleFunctions : FunctionsBase<AdId, NumClicks, AdInput, Output, Empty>
    {
        public override void ReadCompletionCallback(ref AdId key, ref AdInput input, ref Output output, Empty ctx, Status status, RecordMetadata recordMetadata)
        {
            Assert.IsTrue(status.Found);
            Assert.AreEqual(key.adId, output.value.numClicks);
        }

        // Read functions
        public override bool SingleReader(ref AdId key, ref AdInput input, ref NumClicks value, ref Output dst, ref ReadInfo readInfo)
        {
            dst.value = value;
            return true;
        }

        public override bool ConcurrentReader(ref AdId key, ref AdInput input, ref NumClicks value, ref Output dst, ref ReadInfo readInfo)
        {
            dst.value = value;
            return true;
        }

        // RMW functions
        public override bool InitialUpdater(ref AdId key, ref AdInput input, ref NumClicks value, ref Output output, ref RMWInfo rmwInfo) { value = input.numClicks; return true; }

        public override bool InPlaceUpdater(ref AdId key, ref AdInput input, ref NumClicks value, ref Output output, ref RMWInfo rmwInfo)
        {
            Interlocked.Add(ref value.numClicks, input.numClicks.numClicks);
            return true;
        }

        public override bool NeedCopyUpdate(ref AdId key, ref AdInput input, ref NumClicks oldValue, ref Output output, ref RMWInfo rmwInfo) => true;

        public override bool CopyUpdater(ref AdId key, ref AdInput input, ref NumClicks oldValue, ref NumClicks newValue, ref Output output, ref RMWInfo rmwInfo)
        {
            newValue.numClicks += oldValue.numClicks + input.numClicks.numClicks;
            return true;
        }
    }
}
 