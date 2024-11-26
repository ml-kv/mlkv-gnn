﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using FASTER.core;
using NUnit.Framework;
using static FASTER.test.TestUtils;

namespace FASTER.test.Cancellation
{
    [TestFixture]
    class CancellationTests
    {
        internal enum CancelLocation
        {
            None,
            NeedInitialUpdate,
            InitialUpdater,
            NeedCopyUpdate,
            CopyUpdater,
            InPlaceUpdater,
            SingleWriter,
            ConcurrentWriter
        }

        public class CancellationFunctions : FunctionsBase<int, int, int, int, Empty>
        {
            internal CancelLocation cancelLocation = CancelLocation.None;
            internal CancelLocation lastFunc = CancelLocation.None;

            public override bool NeedInitialUpdate(ref int key, ref int input, ref int output, ref RMWInfo rmwInfo)
            {
                lastFunc = CancelLocation.NeedInitialUpdate;
                if (cancelLocation == CancelLocation.NeedInitialUpdate)
                {
                    rmwInfo.Action = RMWAction.CancelOperation;
                    return false;
                }
                return true;
            }

            public override bool NeedCopyUpdate(ref int key, ref int input, ref int oldValue, ref int output, ref RMWInfo rmwInfo)
            {
                lastFunc = CancelLocation.NeedCopyUpdate;
                if (cancelLocation == CancelLocation.NeedCopyUpdate)
                {
                    rmwInfo.Action = RMWAction.CancelOperation;
                    return false;
                }
                return true;
            }

            /// <inheritdoc/>
            public override bool CopyUpdater(ref int key, ref int input, ref int oldValue, ref int newValue, ref int output, ref RMWInfo rmwInfo)
            {
                lastFunc = CancelLocation.CopyUpdater;
                Assert.AreNotEqual(CancelLocation.NeedCopyUpdate, cancelLocation);
                if (cancelLocation == CancelLocation.CopyUpdater)
                {
                    rmwInfo.Action = RMWAction.CancelOperation;
                    return false;
                }
                newValue = oldValue;
                return true;
            }

            public override bool InitialUpdater(ref int key, ref int input, ref int value, ref int output, ref RMWInfo rmwInfo)
            {
                lastFunc = CancelLocation.InitialUpdater;
                Assert.AreNotEqual(CancelLocation.NeedInitialUpdate, cancelLocation);
                Assert.AreNotEqual(CancelLocation.InPlaceUpdater, cancelLocation);
                if (cancelLocation == CancelLocation.InitialUpdater)
                {
                    rmwInfo.Action = RMWAction.CancelOperation;
                    return false;
                }
                value = input;
                return true;
            }

            public override bool InPlaceUpdater(ref int key, ref int input, ref int value, ref int output, ref RMWInfo rmwInfo)
            {
                lastFunc = CancelLocation.InPlaceUpdater;
                if (cancelLocation == CancelLocation.InPlaceUpdater)
                {
                    rmwInfo.Action = RMWAction.CancelOperation;
                    return false;
                }
                value = input;
                return true;
            }

            // Upsert functions
            public override bool SingleWriter(ref int key, ref int input, ref int src, ref int dst, ref int output, ref UpsertInfo upsertInfo, WriteReason reason)
            {
                lastFunc = CancelLocation.SingleWriter;
                if (cancelLocation == CancelLocation.SingleWriter)
                {
                    upsertInfo.Action = UpsertAction.CancelOperation;
                    return false;
                }
                dst = src;
                return true;
            }

            public override bool ConcurrentWriter(ref int key, ref int input, ref int src, ref int dst, ref int output, ref UpsertInfo upsertInfo)
            {
                lastFunc = CancelLocation.ConcurrentWriter;
                if (cancelLocation == CancelLocation.ConcurrentWriter)
                {
                    upsertInfo.Action = UpsertAction.CancelOperation;
                    return false;
                }
                dst = src;
                return true;
            }
        }

        IDevice log;
        CancellationFunctions functions;
        FasterKV<int, int> fht;
        ClientSession<int, int, int, int, Empty, CancellationFunctions> session;

        const int NumRecs = 100;

        [SetUp]
        public void Setup()
        {
            DeleteDirectory(MethodTestDir, wait: true);

            log = Devices.CreateLogDevice(MethodTestDir + "/hlog.log", deleteOnClose: true);
            fht = new FasterKV<int, int>
                (128,
                new LogSettings { LogDevice = log, MemorySizeBits = 17, PageSizeBits = 12 },
                null, null, null, null);

            this.functions = new CancellationFunctions();
            session = fht.For(this.functions).NewSession(this.functions);
        }

        [TearDown]
        public void TearDown()
        {
            session?.Dispose();
            session = null;
            fht?.Dispose();
            fht = null;
            log?.Dispose();
            log = null;
            TestUtils.DeleteDirectory(MethodTestDir);
        }

        private unsafe void Populate()
        {
            // Single alloc outside the loop, to the max length we'll need.
            for (int ii = 0; ii < NumRecs; ii++)
            {
                session.Upsert(ii, ii * NumRecs * 10);
            }
        }

        [Test]
        [Category("FasterKV")]
        [Category("Smoke"), Category("RMW")]
        public void InitialUpdaterTest([Values(Phase.REST, Phase.INTERMEDIATE)] Phase phase)
        {
            Populate();
            session.ctx.phase = phase;
            int key = NumRecs;

            functions.cancelLocation = CancelLocation.NeedInitialUpdate;
            var status = session.RMW(key, key * NumRecs * 10);
            Assert.IsTrue(status.IsCanceled);
            Assert.AreEqual(CancelLocation.NeedInitialUpdate, functions.lastFunc);

            functions.cancelLocation = CancelLocation.InitialUpdater;
            status = session.RMW(key, key * NumRecs * 10);
            Assert.IsTrue(status.IsCanceled);
            Assert.AreEqual(CancelLocation.InitialUpdater, functions.lastFunc);
        }

        [Test]
        [Category("FasterKV")]
        [Category("Smoke"), Category("RMW")]
        public void CopyUpdaterTest([Values(Phase.REST, Phase.INTERMEDIATE)] Phase phase)
        {
            Populate();
            session.ctx.phase = phase;
            int key = NumRecs / 2;

            void do_it()
            {
                for (int lap = 0; lap < 2; ++lap)
                {
                    functions.cancelLocation = CancelLocation.NeedCopyUpdate;
                    var status = session.RMW(key, key * NumRecs * 10);
                    Assert.IsTrue(status.IsCanceled);
                    Assert.AreEqual(CancelLocation.NeedCopyUpdate, functions.lastFunc);

                    functions.cancelLocation = CancelLocation.CopyUpdater;
                    status = session.RMW(key, key * NumRecs * 10);
                    Assert.IsTrue(status.IsCanceled);
                    Assert.AreEqual(CancelLocation.CopyUpdater, functions.lastFunc);
                }
            }

            // First lap tests readonly space
            fht.Log.ShiftReadOnlyAddress(fht.Log.TailAddress, wait: true);
            do_it();

            // Second lap tests OnDisk
            fht.Log.FlushAndEvict(wait: true);
        }

        [Test]
        [Category("FasterKV")]
        [Category("Smoke"), Category("RMW")]
        public void InPlaceUpdaterTest([Values(Phase.REST, Phase.INTERMEDIATE)] Phase phase)
        {
            Populate();
            session.ctx.phase = phase;
            int key = NumRecs / 2;

            // Note: ExpirationTests tests the combination of CancelOperation and DeleteRecord
            functions.cancelLocation = CancelLocation.InPlaceUpdater;
            var status = session.RMW(key, key * NumRecs * 10);
            Assert.IsTrue(status.IsCanceled);
            Assert.AreEqual(CancelLocation.InPlaceUpdater, functions.lastFunc);
        }

        [Test]
        [Category("FasterKV")]
        [Category("Smoke"), Category("RMW")]
        public void SingleWriterTest([Values(Phase.REST, Phase.INTERMEDIATE)] Phase phase)
        {
            Populate();
            session.ctx.phase = phase;
            int key = NumRecs + 1;

            functions.cancelLocation = CancelLocation.SingleWriter;
            var status = session.Upsert(key, key * NumRecs * 10);
            Assert.IsTrue(status.IsCanceled);
            Assert.AreEqual(CancelLocation.SingleWriter, functions.lastFunc);
        }

        [Test]
        [Category("FasterKV")]
        [Category("Smoke"), Category("RMW")]
        public void ConcurrentWriterTest([Values(Phase.REST, Phase.INTERMEDIATE)] Phase phase)
        {
            Populate();
            session.ctx.phase = phase;
            int key = NumRecs / 2;

            functions.cancelLocation = CancelLocation.ConcurrentWriter;
            var status = session.Upsert(key, key * NumRecs * 10);
            Assert.IsTrue(status.IsCanceled);
            Assert.AreEqual(CancelLocation.ConcurrentWriter, functions.lastFunc);
        }
    }
}
