﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.IO;
using FASTER.core;
using FASTER.common;
using Microsoft.Extensions.Logging;

namespace FASTER.server
{
    /// <summary>
    /// FASTER server for variable-length data
    /// </summary>
    public sealed class VarLenServer : IDisposable
    {
        readonly ServerOptions opts;        
        readonly IFasterServer server;
        readonly FasterKV<SpanByte, SpanByte> store;
        readonly SpanByteFasterKVProvider provider;
        readonly SubscribeKVBroker<SpanByte, SpanByte, SpanByte, IKeyInputSerializer<SpanByte, SpanByte>> kvBroker;
        readonly SubscribeBroker<SpanByte, SpanByte, IKeySerializer<SpanByte>> broker;
        readonly LogSettings logSettings;

        /// <summary>
        /// Create server instance; use Start to start the server.
        /// </summary>
        /// <param name="opts"></param>
        /// <param name="loggerFactory"></param>
        public VarLenServer(ServerOptions opts, ILoggerFactory loggerFactory = null)
        {
            this.opts = opts;

            opts.GetSettings(out logSettings, out var checkpointSettings, out var indexSize);
            if (opts.EnableStorageTier && !Directory.Exists(opts.LogDir))
                Directory.CreateDirectory(opts.LogDir);
            if (!Directory.Exists(opts.CheckpointDir))
                Directory.CreateDirectory(opts.CheckpointDir);

            store = new FasterKV<SpanByte, SpanByte>(indexSize, logSettings, checkpointSettings, disableLocking: false, loggerFactory: loggerFactory);

            if (!opts.DisablePubSub)
            {
                kvBroker = new SubscribeKVBroker<SpanByte, SpanByte, SpanByte, IKeyInputSerializer<SpanByte, SpanByte>>(new SpanByteKeySerializer(), null, opts.PubSubPageSizeBytes(), true);
                broker = new SubscribeBroker<SpanByte, SpanByte, IKeySerializer<SpanByte>>(new SpanByteKeySerializer(), null, opts.PubSubPageSizeBytes(), true);
            }

            // Create session provider for VarLen
            provider = new SpanByteFasterKVProvider(store, kvBroker, broker, opts.Recover);

            server = new FasterServerTcp(opts.Address, opts.Port);
            server.Register(WireFormat.DefaultVarLenKV, provider);
            server.Register(WireFormat.WebSocket, provider);
        }

        /// <summary>
        /// Start server instance
        /// </summary>
        public void Start() => server.Start();

        /// <summary>
        /// Dispose store (including log and checkpoint directory)
        /// </summary>
        public void Dispose()
        {
            InternalDispose();
            if (opts.EnableStorageTier && Directory.Exists(opts.LogDir))
                DeleteDirectory(opts.LogDir);
            if (Directory.Exists(opts.CheckpointDir))
                DeleteDirectory(opts.CheckpointDir);
        }

        /// <summary>
        /// Dipose, optionally deleting logs and checkpoints
        /// </summary>
        /// <param name="deleteDir">Whether to delete logs and checkpoints</param>
        public void Dispose(bool deleteDir = true)
        {
            InternalDispose();
            if (deleteDir)
            {
                if (opts.EnableStorageTier && Directory.Exists(opts.LogDir))
                    DeleteDirectory(opts.LogDir);
                if (Directory.Exists(opts.CheckpointDir))
                    DeleteDirectory(opts.CheckpointDir);
            }
        }

        private void InternalDispose()
        {
            server.Dispose();
            broker?.Dispose();
            kvBroker?.Dispose();
            store.Dispose();
            logSettings.LogDevice?.Dispose();
            logSettings.ObjectLogDevice?.Dispose();
        }

        private static void DeleteDirectory(string path)
        {
            if (path == null) return;

            foreach (string directory in Directory.GetDirectories(path))
            {
                DeleteDirectory(directory);
            }

            // Exceptions may happen due to a handle briefly remaining held after Dispose().
            try
            {
                Directory.Delete(path, true);
            }
            catch (Exception ex) when (ex is IOException ||
                                       ex is UnauthorizedAccessException)
            {
                try
                {
                    Directory.Delete(path, true);
                }
                catch { }
            }
        }
    }
}
