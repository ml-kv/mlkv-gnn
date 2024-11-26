// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma warning disable 0162

using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using FASTER.core;

namespace StoreDiskReadBenchmark
{
    public class Program
    {
        static FasterKV<Key, Value> faster;
        static int numOps = 0;

        const int NumParallelSessions = 4; // number of parallel sessions
        const int NumKeys = 20_000_000 / NumParallelSessions; // number of keys in database
        const bool periodicCommit = false; // whether we have a separate periodic commit thread
        const bool waitForCommit = false; // whether upserts wait for commit every so often
        const bool useAsync = false; // whether we use sync or async operations on session
        const bool readBatching = true; // whether we batch reads
        const int readBatchSize = 128; // size of batch, if we are batching reads
        internal const bool simultaneousReadWrite = false; // whether we read & upsert at the same time

        /// <summary>
        /// Main program entry point
        /// </summary>
        static void Main()
        {
            var path = Path.GetTempPath() + "FasterKVDiskReadBenchmark/";
            if (Directory.Exists(path))
                new DirectoryInfo(path).Delete(true);

            // Use real local storage device
            var log = Devices.CreateLogDevice(path + "hlog.log", deleteOnClose: true);

            // Use in-memory device
            // var log = new LocalMemoryDevice(1L << 33, 1L << 30, 1);

            var logSettings = new LogSettings { LogDevice = log, MemorySizeBits = 25, PageSizeBits = 20 };
            var checkpointSettings = new CheckpointSettings { CheckpointDir = path };

            faster = new FasterKV<Key, Value>(1L << 25, logSettings, checkpointSettings);

            ThreadPool.SetMinThreads(2 * Environment.ProcessorCount, 2 * Environment.ProcessorCount);
            TaskScheduler.UnobservedTaskException += (object sender, UnobservedTaskExceptionEventArgs e) =>
            {
                Console.WriteLine($"Unobserved task exception: {e.Exception}");
                e.SetObserved();
            };

            // Threads for reporting, commit
            new Thread(new ThreadStart(ReportThread)).Start();
            if (periodicCommit)
                new Thread(new ThreadStart(CommitThread)).Start();


            Task[] tasks = new Task[NumParallelSessions];
            for (int i = 0; i < NumParallelSessions; i++)
            {
                int local = i;
                tasks[i] = Task.Run(() => AsyncUpsertOperator(local));
            }

            if (!simultaneousReadWrite)
                Task.WaitAll(tasks);

            tasks = new Task[NumParallelSessions];
            for (int i = 0; i < NumParallelSessions; i++)
            {
                int local = i;
                tasks[i] = Task.Run(() => AsyncReadOperator(local));
            }
            Task.WaitAll(tasks);
        }

        /// <summary>
        /// Async upsert operations on FasterKV
        /// </summary>
        static async Task AsyncUpsertOperator(int id)
        {
            using var session = faster.For(new MyFuncs()).NewSession<MyFuncs>(id.ToString() + "upsert");
            await Task.Yield();

            do
            {
                try
                {
                    Key key;
                    Value value;
                    for (int i = NumKeys * id; i < NumKeys * (id + 1); i++)
                    {
                        key = new Key(i);
                        value = new Value(i);
                        session.Upsert(ref key, ref value, Empty.Default, 0);
                        Interlocked.Increment(ref numOps);

                        if (periodicCommit && waitForCommit && i % 100 == 0)
                        {
                            await session.WaitForCommitAsync();
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"{nameof(AsyncUpsertOperator)}({id}): {ex}");
                }
            } while (simultaneousReadWrite);
        }

        /// <summary>
        /// Async read operations on FasterKV
        /// </summary>
        static async Task AsyncReadOperator(int id)
        {
            using var session = faster.For(new MyFuncs()).NewSession<MyFuncs>(id.ToString() + "read");
            Random rand = new(id);

            await Task.Yield();

            try
            {
                Key key;
                Input input = default;
                int i = 0;

                var tasks = new (long, ValueTask<FasterKV<Key, Value>.ReadAsyncResult<Input, Output, Empty>>)[readBatchSize];
                while (true)
                {
                    key = new Key(NumKeys * id + rand.Next(0, NumKeys));

                    if (useAsync)
                    {
                        if (readBatching)
                        {
                            tasks[i % readBatchSize] = (key.key, session.ReadAsync(ref key, ref input));
                        }
                        else
                        {
                            var (status, output) = (await session.ReadAsync(ref key, ref input)).Complete();
                            if (!status.Found || output.value.vfield1 != key.key)
                            {
                                if (!simultaneousReadWrite)
                                    throw new Exception("Wrong value found");
                            }
                        }
                    }
                    else
                    {
                        Output output = new();
                        var result = session.Read(ref key, ref input, ref output, Empty.Default, 0);
                        if (readBatching)
                        {
                            if (!result.IsPending)
                            {
                                if (output.value.vfield1 != key.key)
                                {
                                    if (!simultaneousReadWrite)
                                        throw new Exception("Wrong value found");
                                }
                            }
                        }
                        else
                        {
                            if (result.IsPending)
                            {
                                session.CompletePending(true);
                            }
                            if (output.value.vfield1 != key.key)
                            {
                                if (!simultaneousReadWrite)
                                    throw new Exception("Wrong value found");
                            }
                        }
                    }

                    Interlocked.Increment(ref numOps);
                    i++;

                    if (readBatching && (i % readBatchSize == 0))
                    {
                        if (useAsync)
                        {
                            for (int j = 0; j < readBatchSize; j++)
                            {
                                var (status, output) = (await tasks[j].Item2).Complete();
                                if (!status.Found || output.value.vfield1 != tasks[j].Item1)
                                {
                                    if (!simultaneousReadWrite)
                                        throw new Exception($"Wrong value found. Found: {output.value.vfield1}, Expected: {tasks[j].Item1}");
                                }
                            }
                        }
                        else
                        {
                            session.CompletePending(true);
                        }
                    }

                    if (periodicCommit && waitForCommit && i % 100 == 0)
                    {
                        await session.WaitForCommitAsync();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{nameof(AsyncReadOperator)}({id}): {ex}");
            }
        }

        static void ReportThread()
        {
            long lastTime = 0;
            long lastValue = numOps;

            Stopwatch sw = new();
            sw.Start();

            while (true)
            {
                Thread.Sleep(5000);

                var nowTime = sw.ElapsedMilliseconds;
                var nowValue = numOps;

                Console.WriteLine("Operation Throughput: {0}K ops/sec, Tail: {1}",
                    (nowValue - lastValue) / (double)(nowTime - lastTime), faster.Log.TailAddress);
                lastValue = nowValue;
                lastTime = nowTime;
            }
        }

        static void CommitThread()
        {
            while (true)
            {
                Thread.Sleep(5000);

                // Take log-only checkpoint (quick - no index save)
                faster.TakeHybridLogCheckpointAsync(CheckpointType.FoldOver).GetAwaiter().GetResult();

                // Take index + log checkpoint (longer time)
                // faster.TakeFullCheckpointAsync(CheckpointType.FoldOver).GetAwaiter().GetResult();
            }
        }
    }
}
