﻿using FASTER.core;
using FASTER.server;
using NUnit.Framework;

namespace FASTER.remote.test
{
    [TestFixture]
    public class FixedLenBinaryPubSubTests
    {
        FixedLenServer<long, long, long, long, SimpleFunctions<long, long, long>> server;
        FixedLenClient<long, long> client;

        [SetUp]
        public void Setup()
        {
            server = TestUtils.CreateFixedLenServer(TestContext.CurrentContext.TestDirectory + "/FixedLenBinaryPubSubTests", (a, b) => a + b, disablePubSub: false);
            server.Start();
            client = new FixedLenClient<long, long>();
        }

        [TearDown]
        public void TearDown()
        {
            client.Dispose();
            server.Dispose();
        }

        [Test]
        [Repeat(10)]
        public void SubscribeKVTest()
        {
            var f = new FixedLenClientFunctions();
            using var session = client.GetSession(f);
            using var subSession = client.GetSession(f);

            subSession.SubscribeKV(10);
            subSession.CompletePending(true);
            session.Upsert(10, 23);
            session.CompletePending(true);

            f.WaitSubscribe();
        }

        [Test]
        [Repeat(10)]
        public void PrefixSubscribeKVTest()
        {
            var f = new FixedLenClientFunctions();
            using var session = client.GetSession(f);
            using var subSession = client.GetSession(f);

            subSession.PSubscribeKV(10);
            subSession.CompletePending(true);
            session.Upsert(10, 23);
            session.CompletePending(true);

            f.WaitSubscribe();
        }

        [Test]
        [Repeat(10)]
        public void SubscribeTest()
        {
            var f = new FixedLenClientFunctions();
            using var session = client.GetSession(f);
            using var subSession = client.GetSession(f);

            subSession.Subscribe(10);
            subSession.CompletePending(true);
            session.Publish(10, 23);
            session.CompletePending(true);

            f.WaitSubscribe();
        }

        [Test]
        [Repeat(10)]
        public void PrefixSubscribeTest()
        {
            var f = new FixedLenClientFunctions();
            using var session = client.GetSession(f);
            using var subSession = client.GetSession(f);

            subSession.PSubscribe(10);
            subSession.CompletePending(true);
            session.Publish(10, 23);
            session.CompletePending(true);

            f.WaitSubscribe();
        }
    }
}