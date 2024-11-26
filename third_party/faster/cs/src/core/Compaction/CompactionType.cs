﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

namespace FASTER.core
{
    /// <summary>
    /// Type of log compaction
    /// </summary>
    public enum CompactionType
    {
        /// <summary>
        /// Scan from untilAddress to read-only address to check for record liveness checking
        /// </summary>
        Scan,

        /// <summary>
        /// Lookup each record in compaction range, for record liveness checking using hash chain
        /// </summary>
        Lookup,
    }
}
