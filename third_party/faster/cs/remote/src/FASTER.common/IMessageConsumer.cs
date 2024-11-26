﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;

namespace FASTER.common
{
    /// <summary>
    /// Interface for consumers of messages (from networks), such as sessions
    /// </summary>
    public interface IMessageConsumer : IDisposable
    {
        /// <summary>
        /// Consume the message incoming on the wire
        /// </summary>
        /// <param name="reqBuffer"></param>
        /// <param name="bytesRead"></param>
        /// <returns></returns>
        unsafe int TryConsumeMessages(byte* reqBuffer, int bytesRead);
    }
}
