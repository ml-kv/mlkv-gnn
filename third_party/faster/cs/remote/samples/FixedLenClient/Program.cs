﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using FASTER.client;

namespace FixedLenClient
{
    /// <summary>
    /// Client to interact with FASTER server for fixed-length keys and values
    /// (FixedLenServer). Uses 8-byte keys and 8-byte values.
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_SOCKETS_INLINE_COMPLETIONS", "1");
            string ip = "127.0.0.1";
            int port = 3278;

            if (args.Length > 0 && args[0] != "-")
                ip = args[0];
            if (args.Length > 1 && args[1] != "-")
                port = int.Parse(args[1]);

            // Create a new client, use only blittable struct types here. Client can only communicate
            // with a server that uses the same (or bytewise compatible) blittable struct types. For
            // (long key, long value) is compatible with the FixedLenServer project.
            using var client = new FasterKVClient<long, long>(ip, port);

            // Create a client session to the FasterKV server.
            // Sessions are mono-threaded, similar to normal FasterKV sessions.
            using var session = client.NewSession(new Functions());
            using var session2 = client.NewSession(new Functions());

            // Explicit version of NewSession call, where you provide all types, callback functions, and serializer
            // using var session = client.NewSession<long, long, byte, Functions, FixedLenSerializer<long, long, long, long>>(new Functions(), new FixedLenSerializer<long, long, long, long>());

            // Samples using sync client API
            SyncSamples(session);

            // Samples using sync subscription client API
            SyncSubscriptionSamples(session, session2);

            // Samples using async client API
            AsyncSamples(session).Wait();

            Console.WriteLine("Success!");
        }

        static void SyncSamples(ClientSession<long, long, long, long, byte, Functions, FixedLenSerializer<long, long, long, long>> session)
        {
            session.Upsert(23, 23 + 10000);
            session.CompletePending(true);

            for (int i = 0; i < 1000; i++)
                session.Upsert(i, i + 10000);

            // Flushes partially filled batches, does not wait for responses
            session.Flush();

            // Read key 23, result arrives via ReadCompletionCallback
            session.Read(23);
            session.CompletePending(true);

            // Measure read latency
            double micro = 0;
            for (int i = 0; i < 1000; i++)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                session.Read(23);

                // CompletePending flushes and waits for responses
                // Responses are received on the callback function - see Functions.cs
                session.CompletePending(true);
                sw.Stop();
                if (i > 0)
                    micro += 1000000 * sw.ElapsedTicks / (double)Stopwatch.Frequency;
            }
            Console.WriteLine("Average latency for sync Read: {0} microsecs", micro / (1000-1));

            session.RMW(23, 25);
            session.RMW(23, 25);
            session.CompletePending(true);

            // We use a different context here, to verify the different read result in callback function - see Functions.cs
            session.Read(23, userContext: 1);
            session.CompletePending(true);

            // Now we illustrate Output from RMW directly, again using userContext to control verification - see Functions.cs
            session.RMW(23, 25, userContext: 1);
            session.CompletePending(true);

            for (int i = 100; i < 200; i++)
                session.Upsert(i, i + 10000);

            session.Flush();

            session.CompletePending(true);
        }

        static void SyncSubscriptionSamples(ClientSession<long, long, long, long, byte, Functions, FixedLenSerializer<long, long, long, long>> session, ClientSession<long, long, long, long, byte, Functions, FixedLenSerializer<long, long, long, long>> session2)
        {
            session2.SubscribeKV(23);
            session2.CompletePending(true);

            for (int i = 0; i < 1000000; i++)
                session.Upsert(23, i + 10);

            // Flushes partially filled batches, does not wait for responses
            session.Flush();
            session.CompletePending(true);

            session.RMW(23, 25);
            session.CompletePending(true);

            session.Flush();
            session.CompletePending(true);

            Thread.Sleep(1000);
        }


        static async Task AsyncSamples(ClientSession<long, long, long, long, byte, Functions, FixedLenSerializer<long, long, long, long>> session)
        {
            for (int i = 0; i < 1000; i++)
                session.Upsert(i, i + 10000);

            // By default, we flush async operations as soon as they are issued
            // To instead flush manually, set forceFlush = false in calls below
            var (status, output) = await session.ReadAsync(23);
            if (!status.Found || output != 23 + 10000)
                throw new Exception("Error!");

            // Measure read latency
            double micro = 0;
            for (int i = 0; i < 1000; i++)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                _ = await session.ReadAsync(23);
                sw.Stop();
                if (i > 0)
                    micro += 1000000 * sw.ElapsedTicks / (double)Stopwatch.Frequency;
            }
            Console.WriteLine("Average latency for async Read: {0} microsecs", micro / (1000 - 1));

            await session.DeleteAsync(25);

            long key = 25;
            (status, _) = await session.ReadAsync(key);
            if (!status.NotFound)
                throw new Exception($"Error! Key = {key}; Status = expected NotFound, actual {status}");

            key = 9999;
            (status, _) = await session.ReadAsync(9999);
            if (!status.NotFound)
                throw new Exception($"Error! Key = {key}; Status = expected NotFound, actual {status}");

            key = 9998;
            await session.DeleteAsync(key);

            (status, _) = await session.ReadAsync(9998);
            if (!status.NotFound)
                throw new Exception($"Error! Key = {key}; Status = expected NotFound, actual {status}");

            (status, output) = await session.RMWAsync(9998, 10);
            if (!status.Found || output != 10)
                throw new Exception($"Error! Key = {key}; Status = expected NotFound, actual {status}; output = expected {10}, actual {output}");

            (status, output) = await session.ReadAsync(key);
            if (!status.Found || output != 10)
                throw new Exception($"Error! Key = {key}; Status = expected Found, actual {status}; output = expected {10}, actual {output}");

            (status, output) = await session.RMWAsync(key, 10);
            if (!status.Found || output != 20)
                throw new Exception($"Error! Key = {key}; Status = expected Found, actual {status} output = expected {10}, actual {output}");

            (status, output) = await session.ReadAsync(key);
            if (!status.Found || output != 20)
                throw new Exception($"Error! Key = {key}; Status = expected Found, actual {status}, output = expected {10}, actual {output}");
        }
    }
}
