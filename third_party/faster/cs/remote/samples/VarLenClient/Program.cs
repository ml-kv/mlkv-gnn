﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;

namespace VarLenClient
{
    /// <summary>
    /// FASTER client for variable-length keys and values
    /// Talks to FASTER (VarLenServer) using binary protocol
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

            new MemoryBenchmark().Run(ip, port);
            new MemorySamples().Run(ip, port);
            new CustomTypeSamples().Run(ip, port);
            Console.WriteLine("Success!");
        }
    }
}
