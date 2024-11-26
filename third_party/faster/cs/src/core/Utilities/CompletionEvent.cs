﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System;
using System.Threading;
using System.Threading.Tasks;

namespace FASTER.core
{
    // This structure uses a SemaphoreSlim as if it were a ManualResetEventSlim, because MRES does not support async waiting.
    internal struct CompletionEvent : IDisposable
    {
        private SemaphoreSlim semaphore;

        internal void Initialize() => this.semaphore = new SemaphoreSlim(0);

        internal void Set()
        {
            // If we have an existing semaphore, replace with a new one (to which any subequent waits will apply) and signal any waits on the existing one.
            var newSemaphore = new SemaphoreSlim(0);
            while (true)
            {
                var tempSemaphore = this.semaphore;
                if (tempSemaphore == null)
                {
                    newSemaphore.Dispose();
                    break;
                }
                if (Interlocked.CompareExchange(ref this.semaphore, newSemaphore, tempSemaphore) == tempSemaphore)
                {
                    // Release all waiting threads
                    tempSemaphore.Release(int.MaxValue);
                    break;
                }
            }
        }

        internal bool IsDefault() => this.semaphore is null;

        internal void Wait(CancellationToken token = default) => this.semaphore.Wait(token);

        internal Task WaitAsync(CancellationToken token = default) => this.semaphore.WaitAsync(token);

        /// <inheritdoc/>
        public void Dispose()
        {
            this.semaphore?.Dispose();
            this.semaphore = null;
        }
    }
}

