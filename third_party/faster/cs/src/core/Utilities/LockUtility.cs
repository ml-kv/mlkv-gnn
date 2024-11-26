﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace FASTER.core
{
    internal static class LockUtility
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsIntermediate(this ref RecordInfo recordInfo, out OperationStatus internalStatus, bool isReadingAtAddress = false)
        {
            // First a fast check so there is only one "if"
            internalStatus = OperationStatus.SUCCESS;
            if (!recordInfo.IsIntermediate)
                return false;

            // Separate routine to reduce impact on inlining decision.
            return HandleIntermediate(ref recordInfo, out internalStatus, isReadingAtAddress);
        }

        internal static bool HandleIntermediate(this ref RecordInfo recordInfo, out OperationStatus internalStatus, bool isReadingAtAddress = false)
        {
            // Tentative records have already been allocated.
            while (recordInfo.Tentative)
                Thread.Yield();

            // We don't want to jump out on Sealed/Invalid and restart if we are traversing the "read by address" chain
            if ((recordInfo.Sealed || recordInfo.Invalid) && !isReadingAtAddress)
            {
                Thread.Yield();

                // A Sealed record implies the session that sealed it must make another allocation for a new record, so RETRY_LATER.
                internalStatus = recordInfo.Sealed ? OperationStatus.RETRY_LATER : OperationStatus.RETRY_NOW;
                return true;
            }
            internalStatus = OperationStatus.SUCCESS;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool HandleLockOperation(this ref RecordInfo recordInfo, LockOperation lockOp, out bool isLock)
        {
            isLock = lockOp.LockOperationType == LockOperationType.Lock;

            if (isLock)
                return recordInfo.Lock(lockOp.LockType);

            if (lockOp.LockOperationType == LockOperationType.Unlock)
                recordInfo.Unlock(lockOp.LockType);
            else
                Debug.Fail($"Unexpected LockOperation {lockOp.LockOperationType}");
            return true;
        }

        internal static bool Lock(this ref RecordInfo recordInfo, LockType lockType)
        {
            if (lockType == LockType.Shared)
                return recordInfo.LockShared();
            if (lockType == LockType.Exclusive)
                return recordInfo.LockExclusive();
            if (lockType == LockType.ExclusiveFromShared)
                return recordInfo.LockExclusiveFromShared();
            else
                Debug.Fail($"Unexpected LockType: {lockType}");
            return false;
        }

        internal static void Unlock(this ref RecordInfo recordInfo, LockType lockType)
        {
            if (lockType == LockType.Shared)
                recordInfo.UnlockShared();
            else if (lockType == LockType.Exclusive)
                recordInfo.UnlockExclusive();
            else
                Debug.Fail($"Unexpected LockType: {lockType}");
        }
    }
}
