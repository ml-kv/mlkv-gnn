﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using FASTER.core;
using NUnit.Framework;
using System.Threading;

namespace FASTER.test
{
    [TestFixture]
    internal class FasterLogResumeTests
    {
        private IDevice device;
        private string path;

        [SetUp]
        public void Setup()
        {
            path = TestUtils.MethodTestDir + "/";

            TestUtils.DeleteDirectory(path, wait:true);

            device = Devices.CreateLogDevice(path + "fasterlog.log", deleteOnClose: true);
        }

        [TearDown]
        public void TearDown()
        {
            device?.Dispose();
            device = null;
            TestUtils.DeleteDirectory(path);
        }

        [Test]
        [Category("FasterLog")]
        public async Task FasterLogResumePersistedReaderSpec([Values] LogChecksumType logChecksum)
        {
            CancellationToken cancellationToken = default;

            var input1 = new byte[] { 0, 1, 2, 3 };
            var input2 = new byte[] { 4, 5, 6, 7, 8, 9, 10 };
            var input3 = new byte[] { 11, 12 };
            string readerName = "abc";

            using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum }))
            {
                await l.EnqueueAsync(input1, cancellationToken);
                await l.EnqueueAsync(input2);
                await l.EnqueueAsync(input3);
                await l.CommitAsync();

                using var originalIterator = l.Scan(0, long.MaxValue, readerName);
                Assert.IsTrue(originalIterator.GetNext(out _, out _, out _, out long recoveryAddress));
                originalIterator.CompleteUntil(recoveryAddress);
                Assert.IsTrue(originalIterator.GetNext(out _, out _, out _, out _));  // move the reader ahead
                await l.CommitAsync();
            }

            using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum }))
            {
                using var recoveredIterator = l.Scan(0, long.MaxValue, readerName);
                Assert.IsTrue(recoveredIterator.GetNext(out byte[] outBuf, out _, out _, out _));
                Assert.True(input2.SequenceEqual(outBuf));  // we should have read in input2, not input1 or input3
            }
        }

        [Test]
        [Category("FasterLog")]
        public async Task FasterLogResumeViaCompleteUntilRecordAtSpec([Values] LogChecksumType logChecksum)
        {
            CancellationToken cancellationToken = default;

            var input1 = new byte[] { 0, 1, 2, 3 };
            var input2 = new byte[] { 4, 5, 6, 7, 8, 9, 10 };
            var input3 = new byte[] { 11, 12 };
            string readerName = "abc";

            using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum }))
            {
                await l.EnqueueAsync(input1, cancellationToken);
                await l.EnqueueAsync(input2);
                await l.EnqueueAsync(input3);
                await l.CommitAsync();

                using var originalIterator = l.Scan(0, long.MaxValue, readerName);
                Assert.IsTrue(originalIterator.GetNext(out _, out _, out long recordAddress, out _));
                await originalIterator.CompleteUntilRecordAtAsync(recordAddress);
                Assert.IsTrue(originalIterator.GetNext(out _, out _, out _, out _));  // move the reader ahead
                await l.CommitAsync();
            }

            using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum }))
            {
                using var recoveredIterator = l.Scan(0, long.MaxValue, readerName);
                Assert.IsTrue(recoveredIterator.GetNext(out byte[] outBuf, out _, out _, out _));
                Assert.True(input2.SequenceEqual(outBuf));  // we should have read in input2, not input1 or input3
            }
        }

        [Test]
        [Category("FasterLog")]
        public async Task FasterLogResumePersistedReader2([Values] LogChecksumType logChecksum, [Values] bool removeOutdated)
        {
            var input1 = new byte[] { 0, 1, 2, 3 };
            var input2 = new byte[] { 4, 5, 6, 7, 8, 9, 10 };
            var input3 = new byte[] { 11, 12 };
            string readerName = "abc";

            using (var logCommitManager = new DeviceLogCommitCheckpointManager(new LocalStorageNamedDeviceFactory(), new DefaultCheckpointNamingScheme(path), removeOutdated))
            {
                long originalCompleted;

                using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum, LogCommitManager = logCommitManager }))
                {
                    await l.EnqueueAsync(input1);
                    await l.CommitAsync();
                    await l.EnqueueAsync(input2);
                    await l.CommitAsync();
                    await l.EnqueueAsync(input3);
                    await l.CommitAsync();

                    using var originalIterator = l.Scan(0, long.MaxValue, readerName);
                    Assert.IsTrue(originalIterator.GetNext(out _, out _, out _, out long recoveryAddress));
                    originalIterator.CompleteUntil(recoveryAddress);
                    Assert.IsTrue(originalIterator.GetNext(out _, out _, out _, out _));  // move the reader ahead
                    await l.CommitAsync();
                    originalCompleted = originalIterator.CompletedUntilAddress;
                }

                using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum, LogCommitManager = logCommitManager }))
                {
                    using var recoveredIterator = l.Scan(0, long.MaxValue, readerName);
                    Assert.IsTrue(recoveredIterator.GetNext(out byte[] outBuf, out _, out _, out _));

                    // we should have read in input2, not input1 or input3
                    Assert.True(input2.SequenceEqual(outBuf), $"Original: {input2[0]}, Recovered: {outBuf[0]}, Original: {originalCompleted}, Recovered: {recoveredIterator.CompletedUntilAddress}");

                    // TestContext.Progress.WriteLine($"Original: {originalCompleted}, Recovered: {recoveredIterator.CompletedUntilAddress}"); 
                }
            }
        }

        [Test]
        [Category("FasterLog")]
        public async Task FasterLogResumePersistedReader3([Values] LogChecksumType logChecksum, [Values] bool removeOutdated)
        {
            var input1 = new byte[] { 0, 1, 2, 3 };
            var input2 = new byte[] { 4, 5, 6, 7, 8, 9, 10 };
            var input3 = new byte[] { 11, 12 };
            string readerName = "abcd";

            using (var logCommitManager = new DeviceLogCommitCheckpointManager(new LocalStorageNamedDeviceFactory(), new DefaultCheckpointNamingScheme(path), removeOutdated))
            {
                long originalCompleted;

                using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum, LogCommitManager = logCommitManager }))
                {
                    await l.EnqueueAsync(input1);
                    await l.CommitAsync();
                    await l.EnqueueAsync(input2);
                    await l.CommitAsync();
                    await l.EnqueueAsync(input3);
                    await l.CommitAsync();

                    using var originalIterator = l.Scan(0, l.TailAddress, readerName);

                    int count = 0;
                    await foreach (var item in originalIterator.GetAsyncEnumerable())
                    {
                        if (count < 2) // we complete 1st and 2nd item read
                            originalIterator.CompleteUntil(item.nextAddress);

                        if (count < 1) // we commit only 1st item read
                            await l.CommitAsync();

                        count++;
                    }
                    originalCompleted = originalIterator.CompletedUntilAddress;
                }

                using (var l = new FasterLog(new FasterLogSettings { LogDevice = device, PageSizeBits = 16, MemorySizeBits = 16, LogChecksum = logChecksum, LogCommitManager = logCommitManager }))
                {
                    using var recoveredIterator = l.Scan(0, l.TailAddress, readerName);

                    int count = 0;
                    await foreach (var item in recoveredIterator.GetAsyncEnumerable())
                    {
                        if (count == 0) // resumed iterator will start at item2
                            Assert.True(input2.SequenceEqual(item.entry), $"Original: {input2[0]}, Recovered: {item.entry[0]}, Original: {originalCompleted}, Recovered: {recoveredIterator.CompletedUntilAddress}");
                        count++;
                    }
                    Assert.IsTrue(count == 2);
                }
            }
        }
    }
}
