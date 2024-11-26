﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;

namespace FASTER.core
{
    public partial class FasterKV<Key, Value> : FasterBase, IFasterKV<Key, Value>
    {
        internal struct DeleteAsyncOperation<Input, Output, Context> : IAsyncOperation<Input, Output, Context, DeleteAsyncResult<Input, Output, Context>>
        {
            /// <inheritdoc/>
            public DeleteAsyncResult<Input, Output, Context> CreateCompletedResult(Status status, Output output, RecordMetadata recordMetadata) => new DeleteAsyncResult<Input, Output, Context>(status);

            /// <inheritdoc/>
            public Status DoFastOperation(FasterKV<Key, Value> fasterKV, ref PendingContext<Input, Output, Context> pendingContext, IFasterSession<Key, Value, Input, Output, Context> fasterSession,
                                            FasterExecutionContext<Input, Output, Context> currentCtx, out Output output)
            {
                OperationStatus internalStatus;
                do
                {
                    internalStatus = fasterKV.InternalDelete(ref pendingContext.key.Get(), ref pendingContext.userContext, ref pendingContext, fasterSession, currentCtx, pendingContext.serialNum);
                } while (fasterKV.HandleImmediateRetryStatus(internalStatus, currentCtx, currentCtx, fasterSession, ref pendingContext));
                output = default;
                return TranslateStatus(internalStatus);
            }

            /// <inheritdoc/>
            public ValueTask<DeleteAsyncResult<Input, Output, Context>> DoSlowOperation(FasterKV<Key, Value> fasterKV, IFasterSession<Key, Value, Input, Output, Context> fasterSession,
                                            FasterExecutionContext<Input, Output, Context> currentCtx, PendingContext<Input, Output, Context> pendingContext, CancellationToken token)
                => SlowDeleteAsync(fasterKV, fasterSession, currentCtx, pendingContext, token);

            /// <inheritdoc/>
            public bool HasPendingIO => false;
        }

        /// <summary>
        /// State storage for the completion of an async Delete, or the result if the Delete was completed synchronously
        /// </summary>
        public struct DeleteAsyncResult<Input, Output, Context>
        {
            internal readonly AsyncOperationInternal<Input, Output, Context, DeleteAsyncOperation<Input, Output, Context>, DeleteAsyncResult<Input, Output, Context>> updateAsyncInternal;

            /// <summary>Current status of the Upsert operation</summary>
            public Status Status { get; }

            internal DeleteAsyncResult(Status status)
            {
                this.Status = status;
                this.updateAsyncInternal = default;
            }

            internal DeleteAsyncResult(FasterKV<Key, Value> fasterKV, IFasterSession<Key, Value, Input, Output, Context> fasterSession,
                FasterExecutionContext<Input, Output, Context> currentCtx, PendingContext<Input, Output, Context> pendingContext, ExceptionDispatchInfo exceptionDispatchInfo)
            {
                this.Status = new(StatusCode.Pending);
                updateAsyncInternal = new AsyncOperationInternal<Input, Output, Context, DeleteAsyncOperation<Input, Output, Context>, DeleteAsyncResult<Input, Output, Context>>(
                                        fasterKV, fasterSession, currentCtx, pendingContext, exceptionDispatchInfo, new ());
            }

            /// <summary>Complete the Delete operation, issuing additional allocation asynchronously if needed. It is usually preferable to use Complete() instead of this.</summary>
            /// <returns>ValueTask for Delete result. User needs to await again if result status is Status.PENDING.</returns>
            public ValueTask<DeleteAsyncResult<Input, Output, Context>> CompleteAsync(CancellationToken token = default)
                => this.Status.IsPending
                    ? updateAsyncInternal.CompleteAsync(token)
                    : new ValueTask<DeleteAsyncResult<Input, Output, Context>>(new DeleteAsyncResult<Input, Output, Context>(this.Status));

            /// <summary>Complete the Delete operation, issuing additional I/O synchronously if needed.</summary>
            /// <returns>Status of Delete operation</returns>
            public Status Complete() => this.Status.IsPending ? updateAsyncInternal.CompleteSync().Status : this.Status;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal ValueTask<DeleteAsyncResult<Input, Output, Context>> DeleteAsync<Input, Output, Context>(IFasterSession<Key, Value, Input, Output, Context> fasterSession,
            FasterExecutionContext<Input, Output, Context> currentCtx, ref Key key, Context userContext, long serialNo, CancellationToken token = default)
        {
            var pcontext = new PendingContext<Input, Output, Context> { IsAsync = true };
            return DeleteAsync(fasterSession, currentCtx, ref pcontext, ref key, userContext, serialNo, token);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal ValueTask<DeleteAsyncResult<Input, Output, Context>> DeleteAsync<Input, Output, Context>(IFasterSession<Key, Value, Input, Output, Context> fasterSession,
            FasterExecutionContext<Input, Output, Context> currentCtx, ref PendingContext<Input, Output, Context> pcontext, ref Key key, Context userContext, long serialNo, CancellationToken token)
        {
            fasterSession.UnsafeResumeThread();
            try
            {
                OperationStatus internalStatus;
                do
                {
                    internalStatus = InternalDelete(ref key, ref userContext, ref pcontext, fasterSession, currentCtx, serialNo);
                } while (HandleImmediateRetryStatus(internalStatus, currentCtx, currentCtx, fasterSession, ref pcontext));

                if (OperationStatusUtils.TryConvertToCompletedStatusCode(internalStatus, out Status status))
                    return new ValueTask<DeleteAsyncResult<Input, Output, Context>>(new DeleteAsyncResult<Input, Output, Context>(status));
                Debug.Assert(internalStatus == OperationStatus.ALLOCATE_FAILED);
            }
            finally
            {
                Debug.Assert(serialNo >= currentCtx.serialNum, "Operation serial numbers must be non-decreasing");
                currentCtx.serialNum = serialNo;
                fasterSession.UnsafeSuspendThread();
            }

            return SlowDeleteAsync(this, fasterSession, currentCtx, pcontext, token);
        }

        private static async ValueTask<DeleteAsyncResult<Input, Output, Context>> SlowDeleteAsync<Input, Output, Context>(
            FasterKV<Key, Value> @this,
            IFasterSession<Key, Value, Input, Output, Context> fasterSession,
            FasterExecutionContext<Input, Output, Context> currentCtx,
            PendingContext<Input, Output, Context> pcontext, CancellationToken token = default)
        {
            ExceptionDispatchInfo exceptionDispatchInfo = await WaitForFlushCompletionAsync(@this, pcontext.flushEvent, token).ConfigureAwait(false);
            pcontext.flushEvent = default;
            return new DeleteAsyncResult<Input, Output, Context>(@this, fasterSession, currentCtx, pcontext, exceptionDispatchInfo);
        }
    }
}
