﻿using FASTER.client;
using FASTER.common;
using NUnit.Framework;
using System;
using System.Threading;

namespace FASTER.remote.test
{
    class FixedLenClient<Key, Value> : IDisposable
        where Key : unmanaged
        where Value : unmanaged

    {
        readonly FasterKVClient<long, long> client;

        public FixedLenClient(string address = "127.0.0.1", int port = 33278)
        {
            client = new FasterKVClient<long, long>(address, port);
        }

        public void Dispose()
        {
            client.Dispose();
        }

        public ClientSession<long, long, long, long, long, FixedLenClientFunctions, FixedLenSerializer<long, long, long, long>> GetSession()
            => client.NewSession<long, long, long, FixedLenClientFunctions, FixedLenSerializer<long, long, long, long>>(new FixedLenClientFunctions(), WireFormat.DefaultFixedLenKV);

        public ClientSession<long, long, long, long, long, FixedLenClientFunctions, FixedLenSerializer<long, long, long, long>> GetSession(FixedLenClientFunctions f)
            => client.NewSession<long, long, long, FixedLenClientFunctions, FixedLenSerializer<long, long, long, long>>(f, WireFormat.DefaultFixedLenKV);
    }

    /// <summary>
    /// Callback functions
    /// </summary>
    sealed class FixedLenClientFunctions : CallbackFunctionsBase<long, long, long, long, long>
    {
        readonly ManualResetEvent evt = new(false);

        public override void ReadCompletionCallback(ref long key, ref long input, ref long output, long ctx, Status status)
        {
            Assert.IsTrue(status.Found, status.ToString());
            Assert.IsTrue(output == ctx);
        }

        /// <inheritdoc />
        public override void SubscribeKVCallback(ref long key, ref long input, ref long output, long ctx, Status status)
        {
            Assert.IsTrue(status.Found, status.ToString());
            Assert.IsTrue(output == 23);
            evt.Set();
        }

        /// <inheritdoc />
        public override void SubscribeCallback(ref long key, ref long value, long ctx)
        {
            Assert.IsTrue(value == 23);
            evt.Set();
        }

        public void WaitSubscribe()
        {
            evt.WaitOne();
        }
    }
}
