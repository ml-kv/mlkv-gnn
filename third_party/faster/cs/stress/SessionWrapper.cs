﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using FASTER.core;
using NUnit.Framework;

namespace FASTER.stress
{
    internal class SessionWrapper<TKey, TValue, TInput, TOutput>
    {
        readonly TestLoader testLoader;
        readonly Func<TOutput, int> GetResultKeyOrdinal;
        readonly Random rng;
        readonly Action<TOutput> disposer = o => { };

        ClientSession<TKey, TValue, TInput, TOutput, Empty, IFunctions<TKey, TValue, TInput, TOutput, Empty>> session;
        LockableUnsafeContext<TKey, TValue, TInput, TOutput, Empty, IFunctions<TKey, TValue, TInput, TOutput, Empty>> luContext;

        internal SessionWrapper(TestLoader testLoader, Func<TOutput, int> getResultKeyOrdinal, Random rng, Action<TOutput> disposer = default)
        {
            this.testLoader = testLoader;
            this.GetResultKeyOrdinal = getResultKeyOrdinal;
            this.rng = rng;
            if (disposer is not null)
                this.disposer = disposer;
        }

        internal void PrepareTest(ClientSession<TKey, TValue, TInput, TOutput, Empty, IFunctions<TKey, TValue, TInput, TOutput, Empty>> session)
        {
            this.session = session;
            if (testLoader.WantLUC(rng))
                this.luContext = session.LockableUnsafeContext;
        }

        internal ClientSession<TKey, TValue, TInput, TOutput, Empty, IFunctions<TKey, TValue, TInput, TOutput, Empty>> FkvSession => this.session;

        internal bool IsLUC => !this.luContext.IsNull;

        #region Read

        internal void Read(int keyOrdinal, int keyCount, TKey[] keys)
        {
            if (this.IsLUC)
                ReadLUC(keyOrdinal, keyCount, keys);
            else
                this.Read(keyOrdinal, keys[0]);
        }

        internal Task ReadAsync(int keyOrdinal, int keyCount, TKey[] keys)
            => this.IsLUC ? this.ReadLUCAsync(keyOrdinal, keyCount, keys) : this.ReadAsync(keyOrdinal, keys[0]);

        private void Read(int keyOrdinal, TKey key)
        {
            TOutput output = default;
            var status = session.Read(ref key, ref output);
            if (status.IsPending)
            {
                session.CompletePendingWithOutputs(out var completedOutputs, wait: true);
                (status, output) = TestLoader.GetSinglePendingResult(completedOutputs, out var recordMetadata);
                if (testLoader.UseReadCache)
                    Assert.AreEqual(recordMetadata.Address == Constants.kInvalidAddress, status.Record.CopiedToReadCache, $"keyOrdinal {keyOrdinal}: {status}");
            }
            if (status.Found)
                Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
            else
                Assert.IsTrue(testLoader.UseDelete, status.ToString());
            disposer(output);
        }

        private async Task ReadAsync(int keyOrdinal, TKey key)
        {
            var (status, output) = (await session.ReadAsync(ref key)).Complete();
            if (status.Found)
                Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
            else
                Assert.IsTrue(testLoader.UseDelete, status.ToString());
            disposer(output);
        }

        private void ReadLUC(int keyOrdinal, int keyCount, TKey[] keys)
        {
            try
            {
                luContext.BeginUnsafe();   // Retain epoch control through lock, the operation, and unlock
                testLoader.MaybeLock(luContext, keyCount, keys, isRmw: false, isAsyncTest: false);
                TOutput output = default;
                var status = luContext.Read(ref keys[0], ref output);
                if (status.IsPending)
                {
                    luContext.CompletePendingWithOutputs(out var completedOutputs, wait: true);
                    (status, output) = TestLoader.GetSinglePendingResult(completedOutputs, out var recordMetadata);
                    if (testLoader.UseReadCache)
                        Assert.AreEqual(recordMetadata.Address == Constants.kInvalidAddress, status.Record.CopiedToReadCache, $"keyOrdinal {keyOrdinal}: {status}");
                }
                if (status.Found)
                    Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
                else
                    Assert.IsTrue(testLoader.UseDelete, status.ToString());
                disposer(output);
            }
            finally
            {
                testLoader.MaybeUnlock(luContext, keyCount, keys, isRmw: false, isAsyncTest: false);
                luContext.EndUnsafe();
            }
        }

        private async Task ReadLUCAsync(int keyOrdinal, int keyCount, TKey[] keys)
        {
            try
            {
                testLoader.MaybeLock(luContext, keyCount, keys, isRmw: false, isAsyncTest: true);

                // Do not resume epoch for Async operations
                var (status, output) = (await luContext.ReadAsync(ref keys[0])).Complete();
                if (status.Found)
                    Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
                else
                    Assert.IsTrue(testLoader.UseDelete, status.ToString());
                disposer(output);
            }
            finally
            {
                testLoader.MaybeUnlock(luContext, keyCount, keys, isRmw: false, isAsyncTest: true);
            }
        }
        #endregion Read

        #region RMW
        internal void RMW(int keyOrdinal, int keyCount, TKey[] keys, TInput input)
        {
            if (this.IsLUC)
                this.RMWLUC(keyOrdinal, keyCount, keys, input);
            else
                this.RMW(keyOrdinal, keys[0], input);
        }

        internal Task RMWAsync(int keyOrdinal, int keyCount, TKey[] keys, TInput input)
            => this.IsLUC ? this.RMWLUCAsync(keyOrdinal, keyCount, keys, input) : this.RMWAsync(keyOrdinal, keys[0], input);

        private void RMW(int keyOrdinal, TKey key, TInput input)
        {
            TOutput output = default;
            var status = session.RMW(ref key, ref input, ref output);
            if (status.IsPending)
            {
                session.CompletePendingWithOutputs(out var completedOutputs, wait: true);
                (status, output) = TestLoader.GetSinglePendingResult(completedOutputs, out var recordMetadata);
                Assert.AreEqual(status.Found, status.Record.CopyUpdated | status.Record.InPlaceUpdated, $"keyOrdinal {keyOrdinal}: {status}");
            }
            if (status.Found)
                Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
            else
                Assert.IsTrue(testLoader.UseDelete, status.ToString());
            disposer(output);
        }

        private async Task RMWAsync(int keyOrdinal, TKey key, TInput input)
        {
            var (status, output) = (await session.RMWAsync(ref key, ref input)).Complete();
            if (status.Found)
                Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
            else
                Assert.IsTrue(testLoader.UseDelete, status.ToString());
            disposer(output);
        }

        private void RMWLUC(int keyOrdinal, int keyCount, TKey[] keys, TInput input)
        {
            try
            {
                luContext.BeginUnsafe();   // Retain epoch control through lock, the operation, and unlock
                testLoader.MaybeLock(luContext, keyCount, keys, isRmw: true, isAsyncTest: false);
                TOutput output = default;
                var status = luContext.RMW(ref keys[0], ref input, ref output);
                if (status.IsPending)
                {
                    luContext.CompletePendingWithOutputs(out var completedOutputs, wait: true);
                    (status, output) = TestLoader.GetSinglePendingResult(completedOutputs, out var recordMetadata);
                    Assert.AreEqual(status.Found, status.Record.CopyUpdated | status.Record.InPlaceUpdated, $"keyOrdinal {keyOrdinal}: {status}");
                }
                if (status.Found)
                    Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
                else
                    Assert.IsTrue(testLoader.UseDelete, status.ToString());
                disposer(output);
            }
            finally
            {
                testLoader.MaybeUnlock(luContext, keyCount, keys, isRmw: true, isAsyncTest: false);
                luContext.EndUnsafe();
            }
        }

        private async Task RMWLUCAsync(int keyOrdinal, int keyCount, TKey[] keys, TInput input)
        {
            try
            {
                testLoader.MaybeLock(luContext, keyCount, keys, isRmw: true, isAsyncTest: true);

                // Do not resume epoch for Async operations
                var (status, output) = (await luContext.RMWAsync(ref keys[0], ref input)).Complete();
                if (status.Found)
                    Assert.AreEqual(keyOrdinal, GetResultKeyOrdinal(output));
                else
                    Assert.IsTrue(testLoader.UseDelete, status.ToString());
                disposer(output);
            }
            finally
            {
                testLoader.MaybeUnlock(luContext, keyCount, keys, isRmw: true, isAsyncTest: true);
            }
        }
        #endregion RMW

        #region Upsert
        internal void Upsert(TKey[] keys, TValue value)
        {
            if (this.IsLUC)
                this.UpsertLUC(ref keys[0], ref value);
            else
                this.Upsert(ref keys[0], ref value);
        }

        internal Task UpsertAsync(TKey[] keys, TValue value) => this.IsLUC ? this.UpsertLUCAsync(keys[0], value) : this.UpsertAsync(keys[0], value);

        internal void Upsert(ref TKey key, ref TValue value)
        {
            var status = session.Upsert(ref key, ref value);
            if (status.IsPending)
                session.CompletePending(wait: true);
        }

        private async Task UpsertAsync(TKey key, TValue value) => (await session.UpsertAsync(ref key, ref value)).Complete();

        private void UpsertLUC(ref TKey key, ref TValue value)
        {
            try
            {
                luContext.BeginUnsafe();
                var status = luContext.Upsert(ref key, ref value);
                Assert.IsFalse(status.IsPending, status.ToString());
            }
            finally
            {
                luContext.EndUnsafe();
            }
        }

        private async Task UpsertLUCAsync(TKey key, TValue value) => (await luContext.UpsertAsync(ref key, ref value)).Complete();
        #endregion Upsert

        #region Delete
        internal void Delete(TKey[] keys)
        {
            if (this.IsLUC)
                this.DeleteLUC(keys[0]);
            else
                this.Delete(keys[0]);
        }

        internal Task DeleteAsync(TKey[] keys) => this.IsLUC ? this.DeleteLUCAsync(keys[0]) : this.DeleteAsync(keys[0]);

        private void Delete(TKey key)
        {
            var status = session.Delete(ref key);
            if (status.IsPending)
                session.CompletePending(wait: true);
        }

        private async Task DeleteAsync(TKey key) => (await session.DeleteAsync(ref key)).Complete();

        private void DeleteLUC(TKey key)
        {
            try
            {
                luContext.BeginUnsafe();
                var status = luContext.Delete(ref key);
                Assert.IsFalse(status.IsPending, status.ToString());
            }
            finally
            {
                luContext.EndUnsafe();
            }
        }

        private async Task DeleteLUCAsync(TKey key) => (await luContext.DeleteAsync(ref key)).Complete();

        public void Dispose()
        {
            luContext = default;
            session.Dispose();
            session = default;
        }
        #endregion Delete
    }
}
