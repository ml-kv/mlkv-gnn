﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading;
using FASTER.common;
using FASTER.core;

namespace FASTER.server
{
    /// <summary>
    /// FasterServerBase
    /// </summary>
    public abstract class FasterServerBase : IFasterServer
    {
        /// <summary>
        /// Active sessions
        /// </summary>
        protected readonly ConcurrentDictionary<IMessageConsumer, byte> activeSessions;
        readonly ConcurrentDictionary<WireFormat, ISessionProvider> sessionProviders;
        int activeSessionCount;

        readonly string address;
        readonly int port;
        readonly int networkBufferSize;

        /// <summary>
        /// Server Address
        /// </summary>        
        public string Address => address;

        /// <summary>
        /// Server Port
        /// </summary>
        public int Port => port;

        /// <summary>
        /// Server NetworkBufferSize
        /// </summary>        
        public int NetworkBufferSize => networkBufferSize;

        /// <summary>
        /// Check if server has been disposed
        /// </summary>
        public bool Disposed { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="address"></param>
        /// <param name="port"></param>
        /// <param name="networkBufferSize"></param>
        public FasterServerBase(string address, int port, int networkBufferSize)
        {
            this.address = address;
            this.port = port;
            this.networkBufferSize = networkBufferSize;
            if (networkBufferSize == default)
                this.networkBufferSize = BufferSizeUtils.ClientBufferSize(new MaxSizeSettings());

            activeSessions = new ConcurrentDictionary<IMessageConsumer, byte>();
            sessionProviders = new ConcurrentDictionary<WireFormat, ISessionProvider>();
            activeSessionCount = 0;
            Disposed = false;
        }

        /// <inheritdoc />
        public void Register(WireFormat wireFormat, ISessionProvider backendProvider)
        {
            if (!sessionProviders.TryAdd(wireFormat, backendProvider))
                throw new FasterException($"Wire format {wireFormat} already registered");
        }

        /// <inheritdoc />
        public void Unregister(WireFormat wireFormat, out ISessionProvider provider)
            => sessionProviders.TryRemove(wireFormat, out provider);

        /// <inheritdoc />
        public ConcurrentDictionary<WireFormat, ISessionProvider> GetSessionProviders() => sessionProviders;

        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AddSession(WireFormat protocol, ref ISessionProvider provider, INetworkSender networkSender, out IMessageConsumer session)
        {
            session = null;

            if (Interlocked.Increment(ref activeSessionCount) <= 0)
                return false;

            bool retVal = false;
            try
            {
                session = provider.GetSession(protocol, networkSender);
                retVal = activeSessions.TryAdd(session, default);
                if (!retVal) Interlocked.Decrement(ref activeSessionCount);
            }
            catch
            {
                Interlocked.Decrement(ref activeSessionCount);
            }
            return retVal;
        }

        /// <inheritdoc />
        public abstract void Start();

        /// <inheritdoc />
        public virtual void Dispose()
        {
            Disposed = true;
            DisposeActiveSessions();
            sessionProviders.Clear();
        }

        internal void DisposeActiveSessions()
        {
            while (activeSessionCount >= 0)
            {
                while (activeSessionCount > 0)
                {
                    foreach (var kvp in activeSessions)
                    {
                        var _session = kvp.Key;
                        if (_session != null)
                        {
                            if (activeSessions.TryRemove(_session, out _))
                            {
                                try
                                {
                                    _session.Dispose();
                                }
                                finally
                                {
                                    Interlocked.Decrement(ref activeSessionCount);
                                }
                            }
                        }
                    }
                    Thread.Yield();
                }
                if (Interlocked.CompareExchange(ref activeSessionCount, int.MinValue, 0) == 0)
                    break;
            }
        }

        /// <summary>
        /// Dispose given IServerSession
        /// </summary>
        /// <param name="_session"></param>
        public void DisposeSession(IMessageConsumer _session)
        {
            if (_session != null)
            {
                if (activeSessions.TryRemove(_session, out _))
                {
                    try
                    {
                        _session.Dispose();
                    }
                    finally
                    {
                        Interlocked.Decrement(ref activeSessionCount);
                    }
                }
            }
        }
    }
}
