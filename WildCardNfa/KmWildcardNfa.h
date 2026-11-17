/*
* Windows Kernel Wildcard Matching Engine
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

#pragma once

#include <ntddk.h>
#include <intrin.h>

#define NFA_POOL_TAG 'WcMp'

// -------------------------------------------------------------------------------------
// Template class implementing a lock-free, bit-parallel Non-Deterministic Finite
// Automaton (NFA) capable of evaluating hundreds of wildcard patterns simultaneously
// across both 16-bit Unicode (WCHAR) and 8-bit ANSI/ASCII (CHAR) strings.
//
// Architectural Highlights:
//   1. Compile-Time Trie Prefix Merging: Collapses shared literal and wildcard prefixes
//      across patterns into a unified state DAG, drastically reducing active initial
//      states and eliminating redundant transitions on common path prefixes.
//      Uses Chain-Contiguous Sibling BFS to keep sibling branches contiguous while
//      packing single-child chains into the same 64-bit state word.
//   2. Hierarchical Bitset State Tracking:
//      - <= 63 State Words: Uses stack-allocated state buffers and a single 64-bit
//        word mask register (ulCurrentWordsMask) for zero-allocation O(1) bit scans.
//      - > 63 State Words: Uses a 3-level hierarchy headed by a single 64-bit register
//        (ulSuperMask) tracking up to 64 mask words (4,096 state words = 262,144 states),
//        eliminating linear mask-word loops per character, backed by a lock-free
//        pre-zeroed member scratch buffer with sparse ulSuperMask init/cleanup.
//   3. Register-Direct Single-Pass 4-WCHAR / 8-CHAR SWAR Scanning:
//      - Eliminates stack buffer store/reload roundtrips by loading 8 bytes (4 WCHARs
//        or 8 CHARs) directly into a 64-bit register via Read64Safe, case-folding
//        in-register via SwarUpcaseQuad, and jumping cchOffset directly to the exact
//        matching lane via _BitScanForward64(&ulMatchBit, ullZero).
//   4. Pure 64-Bit SWAR (SIMD Within A Register) Acceleration (No SSE/AVX):
//      - 4-Wide WCHAR / 8-Wide CHAR Parallel ASCII Upcasing: Upcases 4 WCHARs or 8 CHARs
//        simultaneously in a single ULONG64 register using branchless SWAR range arithmetic
//        (IsAsciiSwar + ToUpperSwar).
//      - 4-Wide / 8-Wide Dynamic & Root Steady-State Wildcard ('*') Skipping: Scans
//        CHARS_PER_QUAD characters per step using SWAR zero-lane detection both in the
//        root '*' steady state and dynamically whenever Word 0 is self-sustained by
//        WILD_CARD_SCOPE::SCOPE_DEFAULT Op::Star states (m_ullWord0DefaultStarMask).
//      - Early-Exit on Permanent Accepting States (m_ullWord0PermanentAcceptMask):
//        Precomputes terminal WILD_CARD_SCOPE::SCOPE_DEFAULT Op::Star accepting states 
//        so MatchFirst returns TRUE immediately upon activating an un-killable wildcard
//        state.
//      - 4-Wide / 8-Wide Unbranched Literal-Run & Root Prefix Verification: Compares
//        shared anchored prefixes and unbranched Op::Literal chains anywhere in the Trie
//        8 bytes at a time (using overlapping 64-bit tail loads for unaligned lengths)
//        whenever a single state is active.
//      - 64-Way Bit-Parallel State Filtering: Precomputes per-word 64-bit character
//        compatibility masks (ullAsciiMatchMask[128]) with pre-folded lowercase ASCII
//        slots ('a'..'z') to evaluate all 64 states in a word with a single bitwise AND
//        and zero per-character ASCII upcasing overhead before bit-scanning.
//
// Concurrency & IRQL Contract:
//   - MatchFirst() and MatchAll() are lock-free and safe for concurrent execution across
//     multiple threads after Compile() succeeds.
//   - Mutating methods (AddPattern, Compile, Clear) are not thread-safe with respect to
//     each other or concurrent readers; callers must synchronize mutations externally
//     (e.g., via EX_PUSH_LOCK or rundown protection) if hot-reloading patterns.
//   - Compile() allocates temporary build structures from Paged Pool and requires
//     IRQL <= APC_LEVEL.
//   - MatchFirst() and MatchAll() support IRQL <= DISPATCH_LEVEL for 8-bit CHAR, exact
//     16-bit WCHAR, and pure-ASCII 16-bit WCHAR inputs. Case-insensitive 16-bit WCHAR
//     matching on non-ASCII code points (>= 0x80) requires IRQL < DISPATCH_LEVEL to
//     access paged NLS uppercase tables (gracefully degrades to exact comparison if
//     invoked at >= DISPATCH_LEVEL in release builds).
//
// Template Parameters:
//   TContext - Caller-supplied context type associated with each pattern.
//   TChar    - Character type (WCHAR for 16-bit Unicode, CHAR for 8-bit ANSI/ASCII).
// -------------------------------------------------------------------------------------
template <typename TContext, typename TChar = WCHAR>
class KmWildcardNfa
{
    static_assert(sizeof(TChar) == 1 || sizeof(TChar) == 2,
                  "KmWildcardNfa only supports 8-bit CHAR and 16-bit WCHAR character types.");

    static_assert(__is_trivially_copyable(TContext),
                  "KmWildcardNfa requires TContext to be trivially copyable.");

public:

    // ---------------------------------------------------------------------------------
    // Defines boundary constraints for wildcard operators ('*' and '?').
    // ---------------------------------------------------------------------------------
    enum WILD_CARD_SCOPE : UINT8
    {
        SCOPE_DEFAULT      = 0, // Wildcards can freely cross path separator boundaries
        SCOPE_PATH_SEGMENT = 1  // Wildcards halt expansion upon encountering a path separator
    };

private:

    // ---------------------------------------------------------------------------------
    // Internal VM opcodes representing compiled NFA state transitions.
    // ---------------------------------------------------------------------------------
    enum class Op : UINT8
    {
        Literal = 0, // Matches an exact TChar literal
        Any     = 1, // '?' - Matches any single character within Scope constraints
        Star    = 2, // '*' - Self-looping zero-or-more wildcard within Scope constraints
        Accept  = 3  // Terminal accepting state for patterns not ending in '*'
    };

    // Architectural capacity and stack fast-path constants
    static constexpr ULONG STACK_FAST_PATH_WORDS = 63;                       // Max 64-bit state words (4,032 states) handled on the kernel stack in Path A
    static constexpr ULONG MAX_SUPERMASK_WORDS   = 64;                       // Max 2nd-level mask words tracked by the 64-bit ulSuperMask register
    static constexpr ULONG MAX_STATE_WORDS       = MAX_SUPERMASK_WORDS * 64; // Max 64-bit state words (4,096 words) supported by the 3-level hierarchy
    static constexpr ULONG MAX_INSTRUCTIONS      = MAX_STATE_WORDS * 64;     // Max total NFA states/instructions (262,144 states) supported by the engine

    // High-bit flag stored in NFA_INSTRUCTION::ulExtraOffset when a state starts an
    // unbranched Op::Literal run >= CHARS_PER_QUAD. Safe without masking in the common
    // transition path because unbranched literal-run states have a single child and
    // therefore always have usExtraCount == 0.
    static constexpr ULONG NFA_INST_LITERAL_RUN_FLAG = 0x80000000UL;

    // Character width & 64-bit SWAR lane geometry constants:
    //   - CHAR  (1 byte) : 8 lanes per 64-bit quad, bit-index >> 3 yields lane index 0..7
    //   - WCHAR (2 bytes): 4 lanes per 64-bit quad, bit-index >> 4 yields lane index 0..3
    static constexpr ULONG CHARS_PER_QUAD      = static_cast<ULONG>(sizeof(ULONG64) / sizeof(TChar)); // Number of TChar lanes packed per 64-bit register (8 or 4)
    static constexpr ULONG CHARS_PER_QUAD_MASK = CHARS_PER_QUAD - 1;                                  // Bitmask (7 or 3) for modulo-CHARS_PER_QUAD tail checks
    static constexpr ULONG LANE_BITS_SHIFT     = (sizeof(TChar) == 1) ? 3 : 4;                        // Log2 of bits per lane (3 for 8-bit CHAR, 4 for 16-bit WCHAR)

    // 64-bit SWAR masks parameterized at compile time for 8x8-bit CHAR or 4x16-bit WCHAR
    static constexpr ULONG64 SWAR_ASCII_HIGH_MASK = (sizeof(TChar) == 1) // Masks all non-ASCII bits (>= 0x80) across all lanes in a 64-bit register
        ? 0x8080808080808080ULL
        : 0xFF80FF80FF80FF80ULL;

    static constexpr ULONG64 SWAR_LANE_ONES       = (sizeof(TChar) == 1) // Multiplier containing 1 in the LSB of each lane for broadcasting a TChar
        ? 0x0101010101010101ULL
        : 0x0001000100010001ULL;

    static constexpr ULONG64 SWAR_LANE_MSB_MASK   = (sizeof(TChar) == 1) // Masks the most significant bit (MSB) of each lane for zero-lane detection
        ? 0x8080808080808080ULL
        : 0x8000800080008000ULL;

    // Pattern syntax & path separator character constants for TChar
    static constexpr TChar kStar      = static_cast<TChar>('*');  // Zero-or-more wildcard operator character
    static constexpr TChar kQuestion  = static_cast<TChar>('?');  // Single-character wildcard operator character
    static constexpr TChar kEscape    = static_cast<TChar>('|');  // Escape prefix character for literal '*', '?', and '|'
    static constexpr TChar kBackslash = static_cast<TChar>('\\'); // Path segment separator boundary for SCOPE_PATH_SEGMENT

    // ---------------------------------------------------------------------------------
    // Stores uncompiled pattern metadata added via AddPattern().
    // ---------------------------------------------------------------------------------
    struct PATTERN_ENTRY
    {
        TChar*          pszPattern; // Non-paged pool buffer holding the (optionally case-folded) pattern string
        ULONG           ulLength;   // Length of pszPattern in TChar elements (excluding any null terminator)
        WILD_CARD_SCOPE eScope;     // Boundary behavior for '*' and '?' wildcards within this pattern
        TContext        Context;    // Caller-supplied payload returned when this pattern matches
    };

    // ---------------------------------------------------------------------------------
    // Compile-time Trie node used to deduplicate shared pattern prefixes.
    // ---------------------------------------------------------------------------------
    struct TRIE_NODE
    {
        Op              eOp;             // State transition opcode (Literal, Any, Star, or Accept)
        WILD_CARD_SCOPE eScope;          // Wildcard boundary scope (used for Op::Any and Op::Star)
        TChar           chLiteral;       // Target character code point (used when eOp == Op::Literal)
        ULONG           ulFirstChild;    // Index of first child node in pTrieNodes, or MAXULONG if leaf
        ULONG           ulNextSibling;   // Index of next sibling node under the same parent, or MAXULONG
        ULONG           ulFirstMatchIdx; // Head index in m_pMatchLinks if accepting state, or MAXULONG
        ULONG           ulStateIdx;      // Assigned NFA state index (0..MAX_INSTRUCTIONS-1); transiently stores match list tail during BuildPrefixTrie
    };

    // ---------------------------------------------------------------------------------
    // Singly-linked list node mapping an accepting NFA state directly to one or more
    // caller-supplied pattern contexts (supports identical patterns with distinct IDs).
    // Inlines TContext to eliminate secondary pointer chases into m_pPatterns.
    // ---------------------------------------------------------------------------------
    struct MATCH_LINK_NODE
    {
        TContext Context;        // Inlined caller context for a pattern terminating at this accepting state
        ULONG    ulNextMatchIdx; // Next index in m_pMatchLinks sharing the same accepting state, or MAXULONG
    };

    // ---------------------------------------------------------------------------------
    // Compact 16-byte overflow transition descriptor used when a branching Trie state's
    // epsilon-closure spans across multiple 64-bit state words (4 entries per cache line).
    // ---------------------------------------------------------------------------------
    struct EXTRA_TARGET_WORD
    {
        ULONG64 ullTargetBits; // 64-bit mask of target states to activate within ulTargetWord
        ULONG   ulTargetWord;  // Destination 64-bit state word index (0..m_ulStateWords64-1)
    };

    // ---------------------------------------------------------------------------------
    // Hot 16-byte compiled NFA instruction describing a single state's outgoing
    // epsilon-closed transition bitmask (packs 4 instructions per 64-byte cache line).
    // ---------------------------------------------------------------------------------
    struct NFA_INSTRUCTION
    {
        ULONG64 ullTargetBits; // Primary 64-bit transition bitmask applied to state word usTargetWord
        ULONG   ulExtraOffset; // Base index into m_pExtraTargets; bit 31 stores NFA_INST_LITERAL_RUN_FLAG
        USHORT  usTargetWord;  // Primary destination 64-bit state word index, or MAXUSHORT if dead-end/terminal
        USHORT  usExtraCount;  // Number of additional overflow target words in m_pExtraTargets
    };

    // ---------------------------------------------------------------------------------
    // Cold per-state metadata accessed only during literal-run fast-forwarding,
    // non-ASCII literal filtering, dynamic Word-0 steady-state analysis, or match harvest.
    // ---------------------------------------------------------------------------------
    struct NFA_STATE_METADATA
    {
        ULONG ulFirstMatchIdx;      // Head index into m_pMatchLinks if this state is accepting, else MAXULONG
        ULONG ulLiteralRunOffset;   // Starting character offset of an unbranched literal chain in m_pszLiteralRuns
        ULONG ulLiteralRunLength;   // Remaining length in TChars of the unbranched literal run starting at this state
        ULONG ulLiteralRunEndState; // Final state index reached at the end of the unbranched literal run
        TChar chLiteral;            // Expected literal character (when eOp == Op::Literal)
        Op    eOp;                  // State operation type (Literal, Any, Star, or Accept)
    };

    // ---------------------------------------------------------------------------------
    // Per-64-state-word SWAR filter table allowing all 64 states in a word to be
    // evaluated for character compatibility in a single 64-bit bitwise AND.
    // ---------------------------------------------------------------------------------
    struct WORD_SWAR_FILTER
    {
        ULONG64 ullAsciiMatchMask[128];   // Exact matching state bits for ASCII 0x00..0x7F
        ULONG64 ullNonAsciiWildcardMask;  // Op::Star / Op::Any states with valid targets
        ULONG64 ullNonAsciiLiteralMask;   // Op::Literal states with chLiteral >= 0x80
    };

private:

    // Cache Line 0 (Bytes 0..63): Hot read-only VM execution scalars & core table pointers
    BOOLEAN               m_bCaseInsensitive;            // TRUE if pattern and input matching are case-insensitive
    BOOLEAN               m_bCompiled;                   // Acquire/Release flag indicating runtime NFA tables are valid
    ULONG                 m_ulStateWords64;              // Number of 64-bit state words required for all compiled states
    ULONG                 m_ulPrefixLength;              // Length in TChars of shared anchored root literal prefix (if >= CHARS_PER_QUAD)

    NFA_INSTRUCTION*      m_pInstructions;               // Non-paged array of hot 16-byte compiled state instructions
    NFA_STATE_METADATA*   m_pStateMetadata;              // Non-paged array of cold per-state metadata and literal-run descriptors
    WORD_SWAR_FILTER*     m_pWordFilters;                // Non-paged array of per-word 64-bit SWAR character compatibility filters
    EXTRA_TARGET_WORD*    m_pExtraTargets;               // Non-paged array of overflow transition descriptors for multi-word closures
    MATCH_LINK_NODE*      m_pMatchLinks;                 // Non-paged linked-list array mapping accepting states to pattern contexts
    ULONG64*              m_pulInitialStates;            // Consolidated non-paged buffer: InitialStates[W] + AcceptStates[W] + InitialWordsMask[M]

    // Cache Line 1 (Bytes 64..127): Literal-run table, SWAR 64-bit masks & scratch pointer
    TChar*                m_pszLiteralRuns;              // Non-paged contiguous character buffer storing root prefix and mid-Trie literal runs
    ULONG64               m_ulInitialSuperMask;          // Initial 3rd-level supermask indicating non-zero words in InitialWordsMask
    ULONG64               m_ullSteadyStateWord0;         // Precomputed Word-0 state bitmask for root '*' steady-state skipping (0 if disabled)
    ULONG64               m_ullStarExitBroadcast1;       // First literal exit character broadcast across all lanes of a 64-bit register
    ULONG64               m_ullStarExitBroadcast2;       // Second literal exit character broadcast across all lanes (equals Broadcast1 if only 1 exit)
    ULONG64               m_ullWord0DefaultStarMask;     // Bitmask of Word-0 SCOPE_DEFAULT Op::Star states whose closure stays entirely in Word 0
    ULONG64               m_ullWord0PermanentAcceptMask; // Bitmask of un-killable Word-0 SCOPE_DEFAULT Op::Star accepting states for MatchFirst early-exit
    mutable ULONG64*      m_pulScratchStateBuffer;       // Pre-zeroed non-paged scratch buffer used by Path B (> 63 state words) when uncontended

    // Cache Line 2 (Bytes 128..151): Cold pattern-staging metadata & tail-isolated atomic latch
    PATTERN_ENTRY*        m_pPatterns;                   // Dynamically resized non-paged array of staged pattern entries
    ULONG                 m_ulPatternCount;              // Current number of patterns registered via AddPattern()
    ULONG                 m_ulPatternCapacity;           // Allocated capacity (in entries) of m_pPatterns
    mutable volatile LONG m_lScratchBufferInUse;         // Atomic latch (0 = free, 1 = acquired) guarding m_pulScratchStateBuffer in Path B

private:

    // ---------------------------------------------------------------------------------
    // Returns the number of 64-bit hierarchy mask words required to track m_ulStateWords64.
    // ---------------------------------------------------------------------------------
    __forceinline ULONG GetActiveMaskWords64() const noexcept
    {
        // Divide by 64 rounding up: each 2nd-level mask word tracks 64 state words
        return (m_ulStateWords64 + 63) >> 6;
    }

    // ---------------------------------------------------------------------------------
    // Returns a pointer to the accepting-state bitvector stored immediately after
    // the initial-state bitvector in the consolidated m_pulInitialStates allocation.
    // ---------------------------------------------------------------------------------
    __forceinline ULONG64* GetAcceptStates() const noexcept
    {
        // AcceptStates occupies words [m_ulStateWords64 .. 2*m_ulStateWords64 - 1]
        return m_pulInitialStates + m_ulStateWords64;
    }

    // ---------------------------------------------------------------------------------
    // Returns a pointer to the 2-level initial active word mask array stored after
    // the accepting-state bitvector in the consolidated m_pulInitialStates allocation.
    // ---------------------------------------------------------------------------------
    __forceinline ULONG64* GetInitialWordsMask() const noexcept
    {
        // InitialWordsMask immediately follows InitialStates[W] and AcceptStates[W]
        return m_pulInitialStates + (static_cast<SIZE_T>(m_ulStateWords64) * 2);
    }

    // ---------------------------------------------------------------------------------
    // Zero-extends a TChar to ULONG (and ULONG64) without sign-extension.
    // Prevents signed 8-bit CHAR values >= 0x80 from sign-extending into negative
    // numbers during ASCII table bounds checks or 64-bit SWAR lane shifts/broadcasts.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG CharToUlong(_In_ TChar ch) noexcept
    {
        if constexpr (sizeof(TChar) == 1)
        {
            // Cast via unsigned 8-bit UCHAR first to suppress sign-extension of high ASCII/ANSI bytes
            return static_cast<ULONG>(static_cast<UCHAR>(ch));
        }
        else
        {
            // Cast via unsigned 16-bit USHORT to guarantee zero-extension
            return static_cast<ULONG>(static_cast<USHORT>(ch));
        }
    }

    static __forceinline ULONG64 CharToUint64(_In_ TChar ch) noexcept
    {
        return static_cast<ULONG64>(CharToUlong(ch));
    }

    // ---------------------------------------------------------------------------------
    // Performs an unaligned-safe 64-bit (4-WCHAR or 8-CHAR) load directly into a
    // register. Compiles down to a single 64-bit MOV instruction on x64/ARM64.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG64 Read64Safe(_In_reads_(CHARS_PER_QUAD) const TChar* psz) noexcept
    {
        ULONG64 ullQuad = 0;
        // RtlCopyMemory avoids undefined behavior from unaligned pointer dereferences
        RtlCopyMemory(&ullQuad, psz, sizeof(ULONG64));
        return ullQuad;
    }

    // ---------------------------------------------------------------------------------
    // Fast inline character uppercase helper modeled after KmStringPatternMatchDetail.
    // Differentiates at compile-time between 8-bit CHAR and 16-bit WCHAR:
    //   - 8-bit CHAR : Pure branchless ASCII evaluation, safe at DISPATCH_LEVEL.
    //   - 16-bit WCHAR: Branchless ASCII fast-path for <= 127; falls back to
    //                   RtlUpcaseUnicodeChar for >= 0x80 (requires < DISPATCH_LEVEL).
    // ---------------------------------------------------------------------------------
    static __forceinline TChar UpcaseChar(_In_ TChar ch) noexcept
    {
        if constexpr (sizeof(TChar) == 1)
        {
            // Pure inline branchless evaluation for 8-bit characters; bypasses
            // RtlUpperChar entirely to guarantee safe execution at DISPATCH_LEVEL.
            UCHAR uc = static_cast<UCHAR>(ch);
            return static_cast<TChar>(uc - ((uc >= 'a' && uc <= 'z') << 5));
        }
        else
        {
            // Fast-path branchless ASCII check for 16-bit WCHAR
            USHORT us = static_cast<USHORT>(ch);
            if (us <= 127)
            {
                return static_cast<TChar>(us - ((us >= L'a' && us <= L'z') << 5));
            }

#if DBG
            // Case-insensitive Unicode translation requires NLS tables mapped at < DISPATCH_LEVEL
            ASSERTMSG("Case-insensitive Unicode Match called at DISPATCH_LEVEL or above",
                      KeGetCurrentIrql() < DISPATCH_LEVEL);
#endif
            // Guard against paging faults in release builds if invoked at >= DISPATCH_LEVEL
            if (KeGetCurrentIrql() >= DISPATCH_LEVEL)
            {
                return ch;
            }

            return static_cast<TChar>(RtlUpcaseUnicodeChar(static_cast<WCHAR>(ch)));
        }
    }

    // ---------------------------------------------------------------------------------
    // Checks if an entire 64-bit SWAR block (8 CHARs or 4 WCHARs) is purely ASCII.
    // Enables bypassing per-lane RtlUpcaseUnicodeChar evaluation for pure ASCII text.
    // ---------------------------------------------------------------------------------
    static __forceinline BOOLEAN IsAsciiSwar(_In_ ULONG64 ullQuad) noexcept
    {
        // Any code point >= 0x80 sets at least one bit covered by SWAR_ASCII_HIGH_MASK
        return (ullQuad & SWAR_ASCII_HIGH_MASK) == 0;
    }

    // ---------------------------------------------------------------------------------
    // Case-folds a purely ASCII 64-bit SWAR block (8 CHARs or 4 WCHARs) inline via
    // universal branchless bitwise range arithmetic.
    //
    // By strictly targeting the lower byte of every lane, this routine completely
    // bypasses 16-bit WCHAR carry dependencies. Because the text is guaranteed to be 
    // pure ASCII (IsAsciiSwar == TRUE), the upper byte of a WCHAR is ALWAYS 0x00 and 
    // immune to 0x1F additions breaching the 0x80 bit barrier.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG64 ToUpperSwar(_In_ ULONG64 ullQuad) noexcept
    {
        ULONG64 ullGeA     = ullQuad + 0x1F1F1F1F1F1F1F1FULL; // 0x80 - 'a'
        ULONG64 ullGtZ     = ullQuad + 0x0505050505050505ULL; // 0x80 - '{'
        // Bit 7 of each byte is set iff the byte is >= 'a' (0x61) and <= 'z' (0x7A)
        ULONG64 ullIsLower = (ullGeA & ~ullGtZ) & 0x8080808080808080ULL;

        // Shift the 0x80 bit right by 2 to get 0x20 ('a' - 'A') and XOR to case-fold
        return ullQuad ^ (ullIsLower >> 2);
    }

    // ---------------------------------------------------------------------------------
    // Unrolled slow-path evaluation fallback for SWAR blocks containing non-ASCII code 
    // points. Excluded from inline to preserve L1 cache density in the critical path.
    // ---------------------------------------------------------------------------------
    static __declspec(noinline) ULONG64 SwarUpcaseQuadFallback(_In_ ULONG64 ullQuad) noexcept
    {
        ULONG64 ullFolded = 0;
        
        if constexpr (sizeof(TChar) == 1)
        {
            // Extract, upcase, and repack each of the 8 byte lanes individually
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>(ullQuad & 0xFF))) << 0);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 8) & 0xFF))) << 8);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 16) & 0xFF))) << 16);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 24) & 0xFF))) << 24);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 32) & 0xFF))) << 32);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 40) & 0xFF))) << 40);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 48) & 0xFF))) << 48);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 56) & 0xFF))) << 56);
        }
        else
        {
            // Extract, upcase (via NLS table for >= 0x80), and repack each of the 4 WCHAR lanes
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>(ullQuad & 0xFFFF))) << 0);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 16) & 0xFFFF))) << 16);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 32) & 0xFFFF))) << 32);
            ullFolded |= (CharToUint64(UpcaseChar(static_cast<TChar>((ullQuad >> 48) & 0xFFFF))) << 48);
        }
        
        return ullFolded;
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 1: Upcases a 64-bit block of CHARS_PER_QUAD characters (8 CHARs
    // or 4 WCHARs) simultaneously in-register when all lanes are ASCII.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG64 SwarUpcaseQuad(_In_ ULONG64 ullQuad) noexcept
    {
        // Fast path: branchless in-register case-folding when all lanes are standard ASCII
        if (IsAsciiSwar(ullQuad)) [[likely]]
        {
            return ToUpperSwar(ullQuad);
        }

        // Slow path: per-lane evaluation for blocks containing code points >= 0x80
        return SwarUpcaseQuadFallback(ullQuad);
    }

    // ---------------------------------------------------------------------------------
    // Loads CHARS_PER_QUAD characters (64 bits) from psz and applies SWAR case-folding
    // at compile time when bCaseInsensitive is TRUE.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline ULONG64 ReadNormalizedQuad(_In_reads_(CHARS_PER_QUAD) const TChar* psz) const noexcept
    {
        ULONG64 ullQuad = Read64Safe(psz);
        if constexpr (bCaseInsensitive)
        {
            return SwarUpcaseQuad(ullQuad);
        }
        else
        {
            return ullQuad;
        }
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 1B: Upcases an arbitrary-length TChar buffer into a destination
    // buffer CHARS_PER_QUAD characters (64 bits) at a time during pattern staging.
    // ---------------------------------------------------------------------------------
    static __forceinline VOID SwarUpcaseBuffer(_Out_writes_(cchCount) TChar* __restrict       pDst,
                                               _In_reads_(cchCount)   const TChar* __restrict pSrc,
                                               _In_                   size_t                  cchCount) noexcept
    {
        size_t i = 0;

        // Bulk 64-bit SWAR loop processing CHARS_PER_QUAD characters per iteration
        for (; i + CHARS_PER_QUAD <= cchCount; i += CHARS_PER_QUAD)
        {
            ULONG64 ullQuad = SwarUpcaseQuad(Read64Safe(pSrc + i));
            RtlCopyMemory(pDst + i, &ullQuad, sizeof(ULONG64));
        }

        // Scalar tail loop for remaining (cchCount % CHARS_PER_QUAD) characters
        for (; i < cchCount; i++)
        {
            pDst[i] = UpcaseChar(pSrc[i]);
        }
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 2: Tests whether any of the CHARS_PER_QUAD lanes (8 CHARs or
    // 4 WCHARs) in ullQuad matches the broadcast character in ullBroadcastChar
    // (CharToUint64(ch) * SWAR_LANE_ONES).
    // Returns the 64-bit MSB lane mask (0 if no lane matches; otherwise _BitScanForward64
    // shifted right by LANE_BITS_SHIFT yields the exact 0..(CHARS_PER_QUAD-1) lane index).
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG64 SwarHasChar64(_In_ ULONG64 ullQuad,
                                               _In_ ULONG64 ullBroadcastChar) noexcept
    {
        // XOR produces 0 in any lane matching ullBroadcastChar.
        // Mycroft zero-byte/zero-short detection sets the MSB of the first zero lane.
        ULONG64 ullXor = ullQuad ^ ullBroadcastChar;
        return ((ullXor - SWAR_LANE_ONES) & ~ullXor & SWAR_LANE_MSB_MASK);
    }

    // ---------------------------------------------------------------------------------
    // Shared 64-bit SWAR scanning loop that advances *pcchOffset CHARS_PER_QUAD
    // characters at a time until a lane matches either ullExit1 or ullExit2, landing
    // directly on the first matching character lane.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline VOID SkipUntilDualBroadcastSwar(_In_reads_(cchLen) const TChar* __restrict pszText,
                                                  _In_               size_t                  cchLen,
                                                  _In_               ULONG64                 ullExit1,
                                                  _In_               ULONG64                 ullExit2,
                                                  _Inout_            size_t* __restrict      pcchOffset) const noexcept
    {
        size_t cchOffset = *pcchOffset;

        while (cchOffset + CHARS_PER_QUAD <= cchLen)
        {
            ULONG64 ullQuad = ReadNormalizedQuad<bCaseInsensitive>(pszText + cchOffset);
            ULONG64 ullZero = SwarHasChar64(ullQuad, ullExit1);

            // Check second exit character broadcast if the state has two distinct literal exits
            if (ullExit1 != ullExit2)
            {
                ullZero |= SwarHasChar64(ullQuad, ullExit2);
            }

            // No lane matched either exit character; advance by a full 64-bit quad
            if (ullZero == 0)
            {
                cchOffset += CHARS_PER_QUAD;
                continue;
            }

            // Shifting the matching MSB bit position right by LANE_BITS_SHIFT
            // (>> 3 for 8-bit CHAR, >> 4 for 16-bit WCHAR) yields the exact lane offset.
            ULONG ulMatchBit = 0;
            _BitScanForward64(&ulMatchBit, ullZero);
            cchOffset += (ulMatchBit >> LANE_BITS_SHIFT);
            break;
        }

        *pcchOffset = cchOffset;
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 4: Filters a 64-bit active state word in O(1) time so that only
    // states matching ch and possessing valid outgoing transitions remain set.
    // ---------------------------------------------------------------------------------
    __forceinline ULONG64 FilterActiveStatesSwar(_In_ ULONG   ulWordIdx,
                                                 _In_ ULONG64 ulActiveStates,
                                                 _In_ TChar   ch) const noexcept
    {
        const WORD_SWAR_FILTER& __restrict rFilter = m_pWordFilters[ulWordIdx];
        ULONG                              ulCode  = CharToUlong(ch);

        // O(1) bitmask lookup for all standard ASCII characters (0x00..0x7F)
        if (ulCode < 0x80)
        {
            return ulActiveStates & rFilter.ullAsciiMatchMask[ulCode];
        }

        // For non-ASCII characters (>= 0x80), all active wildcards automatically match
        // (since '\\' is not >= 0x80), and only non-ASCII literals need checking.
        ULONG64 ullResult      = ulActiveStates & rFilter.ullNonAsciiWildcardMask;
        ULONG64 ullLiteralCand = ulActiveStates & rFilter.ullNonAsciiLiteralMask;

        if (ullLiteralCand != 0)
        {
            const NFA_STATE_METADATA* __restrict pStateMetadata = m_pStateMetadata;
            ULONG                                ulBaseStateIdx = ulWordIdx << 6;

            // Iterate over active non-ASCII literal states using Kernighan bit-clearing
            while (ullLiteralCand != 0)
            {
                ULONG ulBit = 0;
                _BitScanForward64(&ulBit, ullLiteralCand);
                ullLiteralCand &= (ullLiteralCand - 1);

                if (pStateMetadata[ulBaseStateIdx + ulBit].chLiteral == ch)
                {
                    ullResult |= (1ULL << ulBit);
                }
            }
        }

        return ullResult;
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 2B: Performs register-direct 64-bit SWAR steady-state skipping
    // when only Word 0 is active. Supports both the precomputed root '*' steady state
    // and dynamic Word-0 steady states sustained by m_ullWord0DefaultStarMask.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline VOID TrySkipWord0SteadyStateSwar(_In_               ULONG64                 ullStates0,
                                                   _In_reads_(cchLen) const TChar* __restrict pszText,
                                                   _In_               size_t                  cchLen,
                                                   _Inout_            size_t* __restrict      pcchOffset) const noexcept
    {
        // Require at least one full 64-bit quad remaining in the input buffer
        if (*pcchOffset + CHARS_PER_QUAD > cchLen)
        {
            return;
        }

        // Fast path: precomputed root '*' steady state (1 or 2 literal exits from root '*').
        // Note: ullStates0 is guaranteed non-zero when Word 0 is active, and m_ullSteadyStateWord0
        // is 0 unless root steady-state skipping is enabled.
        if (ullStates0 == m_ullSteadyStateWord0)
        {
            SkipUntilDualBroadcastSwar<bCaseInsensitive>(pszText,
                                                         cchLen,
                                                         m_ullStarExitBroadcast1,
                                                         m_ullStarExitBroadcast2,
                                                         pcchOffset);
            return;
        }

        // Dynamic path: verify if ullStates0 is self-sustained by Word-0 Default '*' states.
        // A state word is self-sustained if the union of outgoing transitions from its
        // active WILD_CARD_SCOPE::SCOPE_DEFAULT Op::Star states equals ullStates0 itself.
        ULONG64 ullActiveStars = ullStates0 & m_ullWord0DefaultStarMask;
        if (ullActiveStars == 0)
        {
            return;
        }

        const NFA_INSTRUCTION* __restrict    pInstructions  = m_pInstructions;
        const NFA_STATE_METADATA* __restrict pStateMetadata = m_pStateMetadata;
        ULONG64                              ullSustained   = 0;
        ULONG64                              ullTempStars   = ullActiveStars;

        // Accumulate the union of all target states self-sustained by active Word-0 '*' states
        while (ullTempStars != 0)
        {
            ULONG ulStarBit = 0;
            _BitScanForward64(&ulStarBit, ullTempStars);
            ullTempStars &= (ullTempStars - 1);

            ullSustained |= pInstructions[ulStarBit].ullTargetBits;
        }

        // If any currently active state in Word 0 is not regenerated by the '*' loop, abort skip
        if (ullSustained != ullStates0)
        {
            return;
        }

        // Identify non-star active states capable of transitioning outside ullStates0
        ULONG64 ullExitCandidates = ullStates0 & ~ullActiveStars;
        ULONG64 ullExitStateMask  = 0;
        TChar   chExit1           = 0;
        TChar   chExit2           = 0;
        ULONG   ulLiteralExits    = 0;
        BOOLEAN bAllExitLiterals  = TRUE;

        while (ullExitCandidates != 0)
        {
            ULONG ulBit = 0;
            _BitScanForward64(&ulBit, ullExitCandidates);
            ullExitCandidates &= (ullExitCandidates - 1);

            const NFA_INSTRUCTION& rInst = pInstructions[ulBit];
            if (rInst.usTargetWord == MAXUSHORT)
            {
                continue;
            }

            // Ignore states whose transitions remain completely inside ullStates0
            if (rInst.usExtraCount == 0 &&
                rInst.usTargetWord == 0 &&
                (rInst.ullTargetBits & ~ullStates0) == 0)
            {
                continue;
            }

            ullExitStateMask |= (1ULL << ulBit);

            const NFA_STATE_METADATA& rMeta = pStateMetadata[ulBit];
            if (rMeta.eOp == Op::Literal)
            {
                if (ulLiteralExits == 0)
                {
                    chExit1 = rMeta.chLiteral;
                }
                else if (ulLiteralExits == 1)
                {
                    chExit2 = rMeta.chLiteral;
                }

                ulLiteralExits++;
            }
            else
            {
                bAllExitLiterals = FALSE;
            }
        }

        // If no state can transition outside ullStates0, fast-forward to end of buffer
        if (ullExitStateMask == 0)
        {
            *pcchOffset = cchLen;
            return;
        }

        // 1 or 2 literal exit states: use 64-bit SWAR broadcast lane detection
        if (bAllExitLiterals && (ulLiteralExits == 1 || ulLiteralExits == 2))
        {
            ULONG64 ullExit1 = CharToUint64(chExit1) * SWAR_LANE_ONES;
            ULONG64 ullExit2 = (ulLiteralExits == 2)
                ? (CharToUint64(chExit2) * SWAR_LANE_ONES)
                : ullExit1;

            SkipUntilDualBroadcastSwar<bCaseInsensitive>(pszText,
                                                         cchLen,
                                                         ullExit1,
                                                         ullExit2,
                                                         pcchOffset);
            return;
        }

        // > 2 exit states or wildcard exits: 64-bit register load + Word-0 SWAR filter
        // Fully unrolled branchless cascade to preserve zero-overhead L1 pipeline ingestion
        size_t cchOffset = *pcchOffset;

        while (cchOffset + CHARS_PER_QUAD <= cchLen)
        {
            // ASCII lanes (0x00..0x7F) are already case-folded inside ullAsciiMatchMask[128],
            // so ToUpperSwar is bypassed; only non-ASCII WCHAR quads invoke SwarUpcaseQuadFallback.
            ULONG64 ullQuad = Read64Safe(pszText + cchOffset);

            if constexpr (bCaseInsensitive && sizeof(TChar) == 2)
            {
                if (!IsAsciiSwar(ullQuad)) [[unlikely]]
                {
                    ullQuad = SwarUpcaseQuadFallback(ullQuad);
                }
            }
            
            if constexpr (sizeof(TChar) == 1)
            {
                // Test each of the 8 byte lanes against ullExitStateMask via O(1) SWAR filter
                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>(ullQuad & 0xFF)) != 0)
                {
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 8) & 0xFF)) != 0)
                {
                    cchOffset += 1;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 16) & 0xFF)) != 0)
                {
                    cchOffset += 2;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 24) & 0xFF)) != 0)
                {
                    cchOffset += 3;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 32) & 0xFF)) != 0)
                {
                    cchOffset += 4;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 40) & 0xFF)) != 0)
                {
                    cchOffset += 5;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 48) & 0xFF)) != 0)
                {
                    cchOffset += 6;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 56) & 0xFF)) != 0)
                {
                    cchOffset += 7;
                    break;
                }
            }
            else
            {
                // Test each of the 4 WCHAR lanes against ullExitStateMask via O(1) SWAR filter
                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>(ullQuad & 0xFFFF)) != 0)
                {
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 16) & 0xFFFF)) != 0)
                {
                    cchOffset += 1;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 32) & 0xFFFF)) != 0)
                {
                    cchOffset += 2;
                    break;
                }

                if (FilterActiveStatesSwar(0, ullExitStateMask, static_cast<TChar>((ullQuad >> 48) & 0xFFFF)) != 0)
                {
                    cchOffset += 3;
                    break;
                }
            }

            cchOffset += CHARS_PER_QUAD;
        }

        *pcchOffset = cchOffset;
    }

    // ---------------------------------------------------------------------------------
    // Releases all buffers allocated during Compile() while preserving raw patterns.
    // ---------------------------------------------------------------------------------
    VOID FreeCompiledBuffers() noexcept
    {
        // Mark uncompiled with Release semantics before freeing underlying runtime tables
        WriteBooleanRelease(&m_bCompiled, FALSE);

        if (m_pInstructions != NULL)
        {
            ExFreePoolWithTag(m_pInstructions, NFA_POOL_TAG);
            m_pInstructions = NULL;
        }

        if (m_pStateMetadata != NULL)
        {
            ExFreePoolWithTag(m_pStateMetadata, NFA_POOL_TAG);
            m_pStateMetadata = NULL;
        }

        if (m_pWordFilters != NULL)
        {
            ExFreePoolWithTag(m_pWordFilters, NFA_POOL_TAG);
            m_pWordFilters = NULL;
        }

        if (m_pExtraTargets != NULL)
        {
            ExFreePoolWithTag(m_pExtraTargets, NFA_POOL_TAG);
            m_pExtraTargets = NULL;
        }

        if (m_pMatchLinks != NULL)
        {
            ExFreePoolWithTag(m_pMatchLinks, NFA_POOL_TAG);
            m_pMatchLinks = NULL;
        }

        if (m_pulInitialStates != NULL)
        {
            ExFreePoolWithTag(m_pulInitialStates, NFA_POOL_TAG);
            m_pulInitialStates = NULL;
        }

        if (m_pszLiteralRuns != NULL)
        {
            ExFreePoolWithTag(m_pszLiteralRuns, NFA_POOL_TAG);
            m_pszLiteralRuns = NULL;
        }

        if (m_pulScratchStateBuffer != NULL)
        {
            ExFreePoolWithTag(m_pulScratchStateBuffer, NFA_POOL_TAG);
            m_pulScratchStateBuffer = NULL;
        }

        // Reset all compiled execution scalars and precomputed SWAR masks
        m_ulStateWords64              = 0;
        m_ulPrefixLength              = 0;
        m_ulInitialSuperMask          = 0;
        m_ullSteadyStateWord0         = 0;
        m_ullStarExitBroadcast1       = 0;
        m_ullStarExitBroadcast2       = 0;
        m_ullWord0DefaultStarMask     = 0;
        m_ullWord0PermanentAcceptMask = 0;
        m_lScratchBufferInUse         = 0;
    }

    // ---------------------------------------------------------------------------------
    // Recursively computes the epsilon-closure of a Trie node's outgoing transitions.
    // Whenever a child state is Op::Star, it can match zero characters, so its own
    // children are immediately added to the target bitvector as well.
    // ---------------------------------------------------------------------------------
    static VOID ComputeChildrenEpsilonClosure(_In_    const TRIE_NODE* __restrict pTrieNodes,
                                              _In_    ULONG                       ulParentNodeIdx,
                                              _Inout_ ULONG64* __restrict         pulTargetBits) noexcept
    {
        ULONG ulChildIdx = pTrieNodes[ulParentNodeIdx].ulFirstChild;

        while (ulChildIdx != MAXULONG)
        {
            // Map linear state index to its 64-bit word index (>> 6) and bit position (& 63)
            ULONG   ulStateIdx = pTrieNodes[ulChildIdx].ulStateIdx;
            ULONG   ulWordIdx  = ulStateIdx >> 6;
            ULONG64 ullBitMask = 1ULL << (ulStateIdx & 63);

            // Guard against cycles and redundant traversal if already visited
            if ((pulTargetBits[ulWordIdx] & ullBitMask) == 0)
            {
                pulTargetBits[ulWordIdx] |= ullBitMask;

                // Op::Star matches zero or more characters, so propagate epsilon transition to its children
                if (pTrieNodes[ulChildIdx].eOp == Op::Star)
                {
                    ComputeChildrenEpsilonClosure(pTrieNodes,
                                                  ulChildIdx,
                                                  pulTargetBits);
                }
            }

            ulChildIdx = pTrieNodes[ulChildIdx].ulNextSibling;
        }
    }

    // ---------------------------------------------------------------------------------
    // Computes the full outgoing transition bitvector for a single Trie node, including
    // its self-loop bit when the node is Op::Star, plus its children's epsilon-closure.
    // ---------------------------------------------------------------------------------
    VOID ComputeNodeTransitionClosure(_In_                           const TRIE_NODE* __restrict pTrieNodes,
                                      _In_                           ULONG                       ulNodeIdx,
                                      _Out_writes_(m_ulStateWords64) ULONG64* __restrict         pulTargetBits) const noexcept
    {
        RtlZeroMemory(
            pulTargetBits,
            static_cast<SIZE_T>(m_ulStateWords64) * sizeof(ULONG64)
        );

        // Op::Star states re-activate themselves upon consuming a valid character
        if (pTrieNodes[ulNodeIdx].eOp == Op::Star)
        {
            ULONG ulSelfState = pTrieNodes[ulNodeIdx].ulStateIdx;
            pulTargetBits[ulSelfState >> 6] |= (1ULL << (ulSelfState & 63));
        }

        ComputeChildrenEpsilonClosure(pTrieNodes,
                                      ulNodeIdx,
                                      pulTargetBits);
    }

    // ---------------------------------------------------------------------------------
    // Computes the epsilon-closure of a starting Trie node (such as virtual root 0 or
    // the end of a shared root prefix) and populates its 2-level and 3-level hierarchy
    // masks (pulWordsMask and *pulSuperMask).
    // ---------------------------------------------------------------------------------
    VOID ComputeClosureAndHierarchyMasks(_In_                              const TRIE_NODE* __restrict pTrieNodes,
                                         _In_                              ULONG                       ulStartNodeIdx,
                                         _Inout_updates_(m_ulStateWords64) ULONG64* __restrict         pulStates,
                                         _Inout_                           ULONG64* __restrict         pulWordsMask,
                                         _Out_                             ULONG64* __restrict         pulSuperMask) const noexcept
    {
        ComputeChildrenEpsilonClosure(pTrieNodes,
                                      ulStartNodeIdx,
                                      pulStates);

        ULONG64 ullSuper = 0;

        // Roll up non-zero 64-bit state words into 2nd-level (pulWordsMask) and 3rd-level (ullSuper) masks
        for (ULONG w = 0; w < m_ulStateWords64; w++)
        {
            if (pulStates[w] != 0)
            {
                ULONG ulM = w >> 6;
                pulWordsMask[ulM] |= (1ULL << (w & 63));
                ullSuper          |= (1ULL << ulM);
            }
        }

        *pulSuperMask = ullSuper;
    }

    // ---------------------------------------------------------------------------------
    // Walks a maximal unbranched chain of Op::Literal Trie nodes starting at
    // ulStartNodeIdx, stopping before any branching node, non-literal node, or
    // intermediate accepting node. Returns the chain length and sets *pulEndNodeIdx.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG WalkUnbranchedLiteralChain(_In_  const TRIE_NODE* __restrict pTrieNodes,
                                                          _In_  ULONG                       ulStartNodeIdx,
                                                          _Out_ ULONG* __restrict           pulEndNodeIdx) noexcept
    {
        ULONG ulRunLen     = 1;
        ULONG ulCurRunNode = ulStartNodeIdx;

        // Do not fast-forward past intermediate accepting nodes without evaluating them
        while (pTrieNodes[ulCurRunNode].ulFirstMatchIdx == MAXULONG)
        {
            ULONG ulChild = pTrieNodes[ulCurRunNode].ulFirstChild;
            // Halt if leaf reached, node branches into multiple siblings, or child is not a literal
            if (ulChild == MAXULONG ||
                pTrieNodes[ulChild].ulNextSibling != MAXULONG ||
                pTrieNodes[ulChild].eOp != Op::Literal)
            {
                break;
            }

            ulRunLen++;
            ulCurRunNode = ulChild;
        }

        *pulEndNodeIdx = ulCurRunNode;
        return ulRunLen;
    }

    // ---------------------------------------------------------------------------------
    // Searches for an existing Trie child matching (eOp, eScope, chLiteral) under
    // ulParentIdx, or initializes and links a new Trie node if none exists.
    // Returns MAXULONG if allocating a new node would exceed MAX_INSTRUCTIONS states.
    // ---------------------------------------------------------------------------------
    static __forceinline ULONG FindOrInsertTrieChild(_Inout_ TRIE_NODE* __restrict pTrieNodes,
                                                     _Inout_ ULONG& __restrict     rulTrieNodeCount,
                                                     _In_    ULONG                 ulParentIdx,
                                                     _In_    Op                    eOp,
                                                     _In_    WILD_CARD_SCOPE       eScope,
                                                     _In_    TChar                 chLiteral) noexcept
    {
        ULONG ulCur  = pTrieNodes[ulParentIdx].ulFirstChild;
        ULONG ulLast = MAXULONG;

        // Scan sibling list under ulParentIdx to reuse an identical existing prefix transition
        while (ulCur != MAXULONG)
        {
            if (pTrieNodes[ulCur].eOp == eOp &&
                pTrieNodes[ulCur].eScope == eScope &&
                pTrieNodes[ulCur].chLiteral == chLiteral)
            {
                return ulCur;
            }

            ulLast = ulCur;
            ulCur  = pTrieNodes[ulCur].ulNextSibling;
        }

        // Node 0 is the virtual root, so rulTrieNodeCount > MAX_INSTRUCTIONS means all
        // MAX_INSTRUCTIONS state slots (nodes 1..MAX_INSTRUCTIONS) are already in use.
        if (rulTrieNodeCount > MAX_INSTRUCTIONS)
        {
            return MAXULONG;
        }

        ULONG ulNewIdx = rulTrieNodeCount++;

        pTrieNodes[ulNewIdx].eOp             = eOp;
        pTrieNodes[ulNewIdx].eScope          = eScope;
        pTrieNodes[ulNewIdx].chLiteral       = chLiteral;
        pTrieNodes[ulNewIdx].ulFirstChild    = MAXULONG;
        pTrieNodes[ulNewIdx].ulNextSibling   = MAXULONG;
        pTrieNodes[ulNewIdx].ulFirstMatchIdx = MAXULONG;
        pTrieNodes[ulNewIdx].ulStateIdx      = MAXULONG;

        // Link the newly allocated node at the end of the parent's child list
        if (ulLast == MAXULONG)
        {
            pTrieNodes[ulParentIdx].ulFirstChild = ulNewIdx;
        }
        else
        {
            pTrieNodes[ulLast].ulNextSibling = ulNewIdx;
        }

        return ulNewIdx;
    }

    // ---------------------------------------------------------------------------------
    // Allocates the transient prefix Trie and runtime match-link table, inserts all
    // normalized patterns into the Trie, and validates the total state count limit.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(APC_LEVEL)
    NTSTATUS BuildPrefixTrie(_Outptr_ TRIE_NODE** __restrict ppTrieNodes,
                             _Out_    ULONG* __restrict      pulTrieNodeCount) noexcept
    {
        *ppTrieNodes      = NULL;
        *pulTrieNodeCount = 0;

        // Calculate upper bound on Trie nodes: 1 root + sum(pattern length + 1)
        // Promoted to SIZE_T to prevent integer overflow on immense pathological workloads,
        // and capped at (MAX_INSTRUCTIONS + 1) since FindOrInsertTrieChild enforces the
        // MAX_INSTRUCTIONS state limit online during prefix-merging.
        SIZE_T cbMaxTrieNodes = 1;
        for (ULONG i = 0; i < m_ulPatternCount; i++)
        {
            cbMaxTrieNodes += static_cast<SIZE_T>(m_pPatterns[i].ulLength) + 1;
            if (cbMaxTrieNodes > static_cast<SIZE_T>(MAX_INSTRUCTIONS) + 1)
            {
                cbMaxTrieNodes = static_cast<SIZE_T>(MAX_INSTRUCTIONS) + 1;
                break;
            }
        }

        // Allocate the transient Trie on the PAGED pool.
        // We no longer block compilation if cbMaxTrieNodes exceeds MAX_INSTRUCTIONS here, 
        // because Trie prefix-merging will collapse thousands of identical paths down to 
        // a footprint that safely fits inside the engine limits.
        TRIE_NODE* __restrict pTrieNodes = static_cast<TRIE_NODE*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED, 
                                                                                   cbMaxTrieNodes * sizeof(TRIE_NODE),
                                                                                   NFA_POOL_TAG));

        MATCH_LINK_NODE* __restrict pMatchLinks = static_cast<MATCH_LINK_NODE*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED, // Runtime structure accessed in MatchAll; MUST stay NON_PAGED
                                                                                                static_cast<SIZE_T>(m_ulPatternCount) * sizeof(MATCH_LINK_NODE),
                                                                                                NFA_POOL_TAG));

        m_pMatchLinks = pMatchLinks;

        if (pTrieNodes == NULL || pMatchLinks == NULL)
        {
            if (pTrieNodes != NULL)
            {
                ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
            }

            FreeCompiledBuffers();
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        // Initialize virtual Trie root at node 0
        ULONG ulTrieNodeCount          = 1;
        pTrieNodes[0].eOp              = Op::Accept;
        pTrieNodes[0].eScope           = WILD_CARD_SCOPE::SCOPE_DEFAULT;
        pTrieNodes[0].chLiteral        = 0;
        pTrieNodes[0].ulFirstChild     = MAXULONG;
        pTrieNodes[0].ulNextSibling    = MAXULONG;
        pTrieNodes[0].ulFirstMatchIdx  = MAXULONG;
        pTrieNodes[0].ulStateIdx       = MAXULONG;

        // Step 1: Insert all normalized patterns into the prefix Trie
        const PATTERN_ENTRY* __restrict pPatterns = m_pPatterns;

        for (ULONG ulPatIdx = 0; ulPatIdx < m_ulPatternCount; ulPatIdx++)
        {
            const PATTERN_ENTRY& rEntry    = pPatterns[ulPatIdx];
            ULONG                ulCurNode = 0;
            BOOLEAN              bPrevStar = FALSE;

            for (ULONG c = 0; c < rEntry.ulLength; c++)
            {
                TChar ch = rEntry.pszPattern[c];

                // Support '|' as an escape character for literal '*', '?', and '|'
                if (ch == kEscape && c + 1 < rEntry.ulLength)
                {
                    TChar chNext = rEntry.pszPattern[c + 1];
                    
                    if (chNext == kStar || chNext == kQuestion || chNext == kEscape)
                    {
                        c++;
                        ulCurNode = FindOrInsertTrieChild(pTrieNodes,
                                                          ulTrieNodeCount,
                                                          ulCurNode,
                                                          Op::Literal,
                                                          WILD_CARD_SCOPE::SCOPE_DEFAULT,
                                                          chNext);
                        if (ulCurNode == MAXULONG)
                        {
                            ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
                            FreeCompiledBuffers();
                            return STATUS_INSUFFICIENT_RESOURCES;
                        }

                        bPrevStar = FALSE;
                        continue;
                    }
                }

                if (ch == kStar)
                {
                    // Collapse adjacent '*' wildcards into a single self-looping Op::Star state
                    if (!bPrevStar)
                    {
                        ulCurNode = FindOrInsertTrieChild(pTrieNodes,
                                                          ulTrieNodeCount,
                                                          ulCurNode,
                                                          Op::Star,
                                                          rEntry.eScope,
                                                          0);
                        bPrevStar = TRUE;
                    }
                }
                else if (ch == kQuestion)
                {
                    ulCurNode = FindOrInsertTrieChild(pTrieNodes,
                                                      ulTrieNodeCount,
                                                      ulCurNode,
                                                      Op::Any,
                                                      rEntry.eScope,
                                                      0);
                    bPrevStar = FALSE;
                }
                else
                {
                    ulCurNode = FindOrInsertTrieChild(pTrieNodes,
                                                      ulTrieNodeCount,
                                                      ulCurNode,
                                                      Op::Literal,
                                                      WILD_CARD_SCOPE::SCOPE_DEFAULT,
                                                      ch);
                    bPrevStar = FALSE;
                }

                if (ulCurNode == MAXULONG)
                {
                    ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
                    FreeCompiledBuffers();
                    return STATUS_INSUFFICIENT_RESOURCES;
                }
            }

            // Patterns not ending in '*' require a terminal Op::Accept state so they only
            // match when the end of the input string is reached.
            if (!bPrevStar)
            {
                ulCurNode = FindOrInsertTrieChild(pTrieNodes,
                                                  ulTrieNodeCount,
                                                  ulCurNode,
                                                  Op::Accept,
                                                  WILD_CARD_SCOPE::SCOPE_DEFAULT,
                                                  0);
                if (ulCurNode == MAXULONG)
                {
                    ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
                    FreeCompiledBuffers();
                    return STATUS_INSUFFICIENT_RESOURCES;
                }
            }

            // Each pattern index maps 1:1 to its own MATCH_LINK_NODE entry with inlined Context.
            // Append in FIFO registration order in O(1) using ulStateIdx as a transient tail
            // cursor (restored to MAXULONG below before AssignStateIndicesBfs).
            pMatchLinks[ulPatIdx].Context        = rEntry.Context;
            pMatchLinks[ulPatIdx].ulNextMatchIdx = MAXULONG;

            if (pTrieNodes[ulCurNode].ulFirstMatchIdx == MAXULONG)
            {
                pTrieNodes[ulCurNode].ulFirstMatchIdx = ulPatIdx;
            }
            else
            {
                ULONG ulTailIdx = pTrieNodes[ulCurNode].ulStateIdx;
                pMatchLinks[ulTailIdx].ulNextMatchIdx = ulPatIdx;
            }

            pTrieNodes[ulCurNode].ulStateIdx = ulPatIdx;
        }

        // Restore ulStateIdx sentinel before returning the Trie to Compile()/AssignStateIndicesBfs
        for (ULONG i = 0; i < ulTrieNodeCount; i++)
        {
            pTrieNodes[i].ulStateIdx = MAXULONG;
        }

        // Enforce the strict runtime limit AFTER Trie deduplication has occurred.
        // Note: ulTotalStates is guaranteed >= 1 because m_ulPatternCount > 0 and every
        // pattern inserts at least one Trie child node (Op::Star, Op::Any, Op::Literal, or Op::Accept).
        ULONG ulTotalStates = ulTrieNodeCount - 1;
        if (ulTotalStates > MAX_INSTRUCTIONS)
        {
            ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
            FreeCompiledBuffers();
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        *ppTrieNodes      = pTrieNodes;
        *pulTrieNodeCount = ulTrieNodeCount;
        return STATUS_SUCCESS;
    }

    // ---------------------------------------------------------------------------------
    // Assigns state indices using Chain-Contiguous Sibling BFS so sibling branches
    // remain contiguous while single-child chains pack into the same 64-bit state word.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(APC_LEVEL)
    static NTSTATUS AssignStateIndicesBfs(_Inout_updates_(ulTrieNodeCount) TRIE_NODE* __restrict pTrieNodes,
                                          _In_                             ULONG                 ulTrieNodeCount) noexcept
    {
        // Step 2: Assign state indices using Chain-Contiguous Sibling BFS so sibling
        // branches remain contiguous while single-child chains pack into the same word.
        ULONG* __restrict pulBfsQueue = static_cast<ULONG*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED, // PAGED for temporary queue allocation
                                                                            static_cast<SIZE_T>(ulTrieNodeCount) * sizeof(ULONG),
                                                                            NFA_POOL_TAG));

        if (pulBfsQueue == NULL)
        {
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        ULONG ulQueueHead = 0;
        ULONG ulQueueTail = 0;
        ULONG ulNextState = 0;

        // Seed the BFS queue with virtual root node 0 (which does not consume a state index)
        pulBfsQueue[ulQueueTail++] = 0;

        while (ulQueueHead < ulQueueTail)
        {
            ULONG ulNodeIdx = pulBfsQueue[ulQueueHead++];

            // Pack single-child chains contiguously into the same 64-bit state word
            while (pTrieNodes[ulNodeIdx].ulFirstChild != MAXULONG &&
                   pTrieNodes[pTrieNodes[ulNodeIdx].ulFirstChild].ulNextSibling == MAXULONG)
            {
                ULONG ulSingleChild                  = pTrieNodes[ulNodeIdx].ulFirstChild;
                pTrieNodes[ulSingleChild].ulStateIdx = ulNextState++;
                ulNodeIdx                            = ulSingleChild;
            }

            // Assign remaining sibling branches contiguously before queueing
            ULONG ulChildIdx = pTrieNodes[ulNodeIdx].ulFirstChild;
            while (ulChildIdx != MAXULONG)
            {
                pTrieNodes[ulChildIdx].ulStateIdx = ulNextState++;
                pulBfsQueue[ulQueueTail++]        = ulChildIdx;
                ulChildIdx                        = pTrieNodes[ulChildIdx].ulNextSibling;
            }
        }

        ExFreePoolWithTag(pulBfsQueue, NFA_POOL_TAG);
        return STATUS_SUCCESS;
    }

    // ---------------------------------------------------------------------------------
    // Allocates runtime NFA instruction, state metadata, SWAR filter, consolidated
    // bitvectors, scratch, and overflow target tables sized to the deduplicated Trie.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(APC_LEVEL)
    NTSTATUS AllocateRuntimeTables(_In_reads_(ulTrieNodeCount) const TRIE_NODE* __restrict pTrieNodes,
                                   _In_                        ULONG                       ulTrieNodeCount,
                                   _Outptr_                    ULONG64** __restrict        ppulScratchTargetBits) noexcept
    {
        *ppulScratchTargetBits = NULL;

        // Step 3: Allocate runtime NFA instruction, cold state metadata, and SWAR filter tables
        ULONG  ulInstructionCount  = ulTrieNodeCount - 1;
        m_ulStateWords64           = (ulInstructionCount + 63) / 64;
        ULONG  ulActiveMaskWords64 = GetActiveMaskWords64();
        SIZE_T cbBitvectorWords    = (static_cast<SIZE_T>(m_ulStateWords64) * 2) + ulActiveMaskWords64;

        m_pInstructions = static_cast<NFA_INSTRUCTION*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                                        static_cast<SIZE_T>(ulInstructionCount) * sizeof(NFA_INSTRUCTION),
                                                                        NFA_POOL_TAG));

        m_pStateMetadata = static_cast<NFA_STATE_METADATA*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                                            static_cast<SIZE_T>(ulInstructionCount) * sizeof(NFA_STATE_METADATA),
                                                                            NFA_POOL_TAG));

        m_pWordFilters = static_cast<WORD_SWAR_FILTER*>(ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                                        static_cast<SIZE_T>(m_ulStateWords64) * sizeof(WORD_SWAR_FILTER),
                                                                        NFA_POOL_TAG));

        // Consolidate InitialStates (W words), AcceptStates (W words), and InitialWordsMask (M words)
        // into a single contiguous zero-initialized Non-Paged Pool allocation.
        m_pulInitialStates = static_cast<ULONG64*>(ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                                   cbBitvectorWords * sizeof(ULONG64),
                                                                   NFA_POOL_TAG));

        ULONG64* __restrict pulScratchTargetBits = static_cast<ULONG64*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED, // PAGED for temporary bitvector allocation
                                                                                         static_cast<SIZE_T>(m_ulStateWords64) * sizeof(ULONG64),
                                                                                         NFA_POOL_TAG));

        if (m_pInstructions == NULL ||
            m_pStateMetadata == NULL ||
            m_pWordFilters == NULL ||
            m_pulInitialStates == NULL ||
            pulScratchTargetBits == NULL)
        {
            if (pulScratchTargetBits != NULL)
            {
                ExFreePoolWithTag(pulScratchTargetBits, NFA_POOL_TAG);
            }

            return STATUS_INSUFFICIENT_RESOURCES;
        }

        // Pre-allocate zeroed member scratch buffer for > 63 state word path
        if (m_ulStateWords64 > STACK_FAST_PATH_WORDS)
        {
            // Double-buffered layout: CurrentStates[W] + NextStates[W] + CurrentWordsMask[M] + NextWordsMask[M]
            SIZE_T cbScratchWords   = static_cast<SIZE_T>(m_ulStateWords64 + ulActiveMaskWords64) * 2;
            m_pulScratchStateBuffer = static_cast<ULONG64*>(ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                                            cbScratchWords * sizeof(ULONG64),
                                                                            NFA_POOL_TAG));
            if (m_pulScratchStateBuffer == NULL)
            {
                ExFreePoolWithTag(pulScratchTargetBits, NFA_POOL_TAG);
                return STATUS_INSUFFICIENT_RESOURCES;
            }
        }

        // Count overflow target words for states whose target closure spans > 1 state word
        ULONG ulTotalExtraWords = 0;

        for (ULONG ulNodeIdx = 1; ulNodeIdx < ulTrieNodeCount; ulNodeIdx++)
        {
            ComputeNodeTransitionClosure(pTrieNodes, ulNodeIdx, pulScratchTargetBits);

            ULONG ulWordsUsed = 0;
            for (ULONG w = 0; w < m_ulStateWords64; w++)
            {
                if (pulScratchTargetBits[w] != 0)
                {
                    ulWordsUsed++;
                }
            }

            // First non-zero target word fits inline in NFA_INSTRUCTION; remaining words spill into m_pExtraTargets
            if (ulWordsUsed > 1)
            {
                ulTotalExtraWords += (ulWordsUsed - 1);
            }
        }

        if (ulTotalExtraWords > 0)
        {
            m_pExtraTargets = static_cast<EXTRA_TARGET_WORD*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                                              static_cast<SIZE_T>(ulTotalExtraWords) * sizeof(EXTRA_TARGET_WORD),
                                                                              NFA_POOL_TAG));
            if (m_pExtraTargets == NULL)
            {
                ExFreePoolWithTag(pulScratchTargetBits, NFA_POOL_TAG);
                return STATUS_INSUFFICIENT_RESOURCES;
            }
        }

        *ppulScratchTargetBits = pulScratchTargetBits;
        return STATUS_SUCCESS;
    }

    // ---------------------------------------------------------------------------------
    // Populates compiled instructions, cold state metadata, overflow transition targets,
    // per-word SWAR character filter masks, and initial active state closures.
    // ---------------------------------------------------------------------------------
    VOID PopulateInstructionsAndFilters(_In_reads_(ulTrieNodeCount)       const TRIE_NODE* __restrict pTrieNodes,
                                        _In_                              ULONG                       ulTrieNodeCount,
                                        _Inout_updates_(m_ulStateWords64) ULONG64* __restrict         pulScratchTargetBits) noexcept
    {
        // Populate hot 16-byte instructions, cold state metadata, and per-word SWAR filter masks
        ULONG                         ulExtraCursor   = 0;
        ULONG64* __restrict           pulAcceptStates = GetAcceptStates();
        NFA_INSTRUCTION* __restrict   pInstructions   = m_pInstructions;
        NFA_STATE_METADATA* __restrict pStateMetadata = m_pStateMetadata;
        WORD_SWAR_FILTER* __restrict  pWordFilters    = m_pWordFilters;
        EXTRA_TARGET_WORD* __restrict pExtraTargets   = m_pExtraTargets;

        for (ULONG ulNodeIdx = 1; ulNodeIdx < ulTrieNodeCount; ulNodeIdx++)
        {
            ULONG               ulStateIdx = pTrieNodes[ulNodeIdx].ulStateIdx;
            NFA_INSTRUCTION&    rInst      = pInstructions[ulStateIdx];
            NFA_STATE_METADATA& rMeta      = pStateMetadata[ulStateIdx];
            Op                  eNodeOp    = pTrieNodes[ulNodeIdx].eOp;
            WILD_CARD_SCOPE     eNodeScope = pTrieNodes[ulNodeIdx].eScope;
            TChar               chLiteral  = pTrieNodes[ulNodeIdx].chLiteral;

            rInst.ullTargetBits        = 0;
            rInst.ulExtraOffset        = ulExtraCursor;
            rInst.usTargetWord         = MAXUSHORT;
            rInst.usExtraCount         = 0;

            rMeta.ulFirstMatchIdx      = pTrieNodes[ulNodeIdx].ulFirstMatchIdx;
            rMeta.ulLiteralRunOffset   = 0;
            rMeta.ulLiteralRunLength   = 0;
            rMeta.ulLiteralRunEndState = ulStateIdx;
            rMeta.chLiteral            = chLiteral;
            rMeta.eOp                  = eNodeOp;

            // Mark accepting state bitvector and check if it qualifies as a permanent Word-0 accepting '*' state
            if (rMeta.ulFirstMatchIdx != MAXULONG)
            {
                pulAcceptStates[ulStateIdx >> 6] |= (1ULL << (ulStateIdx & 63));

                // A SCOPE_DEFAULT Op::Star state that is also accepting can never be killed by subsequent input
                if ((ulStateIdx >> 6) == 0 &&
                    eNodeOp == Op::Star &&
                    eNodeScope == WILD_CARD_SCOPE::SCOPE_DEFAULT)
                {
                    m_ullWord0PermanentAcceptMask |= (1ULL << (ulStateIdx & 63));
                }
            }

            ComputeNodeTransitionClosure(pTrieNodes, ulNodeIdx, pulScratchTargetBits);

            BOOLEAN bPrimarySet = FALSE;

            // Store the first non-zero target state word inline in rInst and spill any additional words to m_pExtraTargets
            for (ULONG w = 0; w < m_ulStateWords64; w++)
            {
                ULONG64 ullBits = pulScratchTargetBits[w];
                if (ullBits != 0)
                {
                    if (!bPrimarySet)
                    {
                        rInst.usTargetWord  = static_cast<USHORT>(w);
                        rInst.ullTargetBits = ullBits;
                        bPrimarySet         = TRUE;
                    }
                    else
                    {
                        EXTRA_TARGET_WORD& rExtra = pExtraTargets[ulExtraCursor++];
                        rExtra.ullTargetBits      = ullBits;
                        rExtra.ulTargetWord       = w;
                        rInst.usExtraCount++;
                    }
                }
            }

            // Record Word-0 WILD_CARD_SCOPE::SCOPE_DEFAULT Op::Star states whose closure stays in Word 0
            if ((ulStateIdx >> 6) == 0 &&
                eNodeOp == Op::Star &&
                eNodeScope == WILD_CARD_SCOPE::SCOPE_DEFAULT &&
                rInst.usTargetWord == 0 &&
                rInst.usExtraCount == 0)
            {
                m_ullWord0DefaultStarMask |= (1ULL << (ulStateIdx & 63));
            }

            // Populate SWAR 64-state word compatibility masks for states with valid targets
            if (rInst.usTargetWord != MAXUSHORT)
            {
                ULONG             ulWordIdx   = ulStateIdx >> 6;
                ULONG64           ullStateBit = 1ULL << (ulStateIdx & 63);
                WORD_SWAR_FILTER& rFilter     = pWordFilters[ulWordIdx];

                if (eNodeOp == Op::Literal)
                {
                    ULONG ulCode = CharToUlong(chLiteral);
                    if (ulCode < 0x80)
                    {
                        rFilter.ullAsciiMatchMask[ulCode] |= ullStateBit;

                        // Pre-populate lowercase ASCII slot ('a'..'z') so FilterActiveStatesSwar is natively
                        // case-insensitive for 0x00..0x7F without calling UpcaseChar() per character.
                        if (m_bCaseInsensitive && ulCode >= 'A' && ulCode <= 'Z')
                        {
                            rFilter.ullAsciiMatchMask[ulCode + ('a' - 'A')] |= ullStateBit;
                        }
                    }
                    else
                    {
                        rFilter.ullNonAsciiLiteralMask |= ullStateBit;
                    }
                }
                else if (eNodeOp == Op::Any || eNodeOp == Op::Star)
                {
                    rFilter.ullNonAsciiWildcardMask |= ullStateBit;

                    for (ULONG ch = 0; ch < 128; ch++)
                    {
                        // SCOPE_PATH_SEGMENT wildcards cannot match or cross backslash separators
                        if (eNodeScope == WILD_CARD_SCOPE::SCOPE_PATH_SEGMENT &&
                            ch == static_cast<ULONG>(kBackslash))
                        {
                            continue;
                        }

                        rFilter.ullAsciiMatchMask[ch] |= ullStateBit;
                    }
                }
            }
        }

        // Compute initial active states (epsilon-closure of virtual root node 0)
        ComputeClosureAndHierarchyMasks(pTrieNodes,
                                        0,
                                        m_pulInitialStates,
                                        GetInitialWordsMask(),
                                        &m_ulInitialSuperMask);
    }

    // ---------------------------------------------------------------------------------
    // Configures SWAR Optimization 2 (Steady-State '*' 64-Bit Fast-Forward) when the
    // virtual root has a single default-scope '*' child with 1 or 2 literal exits.
    // ---------------------------------------------------------------------------------
    VOID ConfigureStarSteadyStateSkip(_In_ const TRIE_NODE* __restrict pTrieNodes) noexcept
    {
        // Step 4: Configure SWAR Optimization 2 (Steady-State '*' 64-Bit Fast-Forward)
        ULONG ulRootChild = pTrieNodes[0].ulFirstChild;
        if (ulRootChild != MAXULONG &&
            pTrieNodes[ulRootChild].ulNextSibling == MAXULONG &&
            pTrieNodes[ulRootChild].eOp == Op::Star &&
            pTrieNodes[ulRootChild].eScope == WILD_CARD_SCOPE::SCOPE_DEFAULT &&
            m_ulInitialSuperMask == 1ULL &&
            GetInitialWordsMask()[0] == 1ULL)
        {
            TChar   chExit1      = 0;
            TChar   chExit2      = 0;
            ULONG   ulExitCount  = 0;
            BOOLEAN bCanSkipQuad = TRUE;

            // Inspect all children of the root '*' node to ensure exits are solely 1 or 2 literals
            ULONG ulStarChild = pTrieNodes[ulRootChild].ulFirstChild;
            while (ulStarChild != MAXULONG)
            {
                if (pTrieNodes[ulStarChild].eOp == Op::Literal)
                {
                    if (ulExitCount == 0)
                    {
                        chExit1 = pTrieNodes[ulStarChild].chLiteral;
                    }
                    else if (ulExitCount == 1)
                    {
                        chExit2 = pTrieNodes[ulStarChild].chLiteral;
                    }

                    ulExitCount++;
                }
                else if (pTrieNodes[ulStarChild].eOp != Op::Accept)
                {
                    bCanSkipQuad = FALSE;
                    break;
                }

                ulStarChild = pTrieNodes[ulStarChild].ulNextSibling;
            }

            // Broadcast the 1 or 2 literal exit characters across all lanes of a 64-bit register
            if (bCanSkipQuad && (ulExitCount == 1 || ulExitCount == 2))
            {
                m_ullSteadyStateWord0   = m_pulInitialStates[0];
                m_ullStarExitBroadcast1 = CharToUint64(chExit1) * SWAR_LANE_ONES;
                m_ullStarExitBroadcast2 = (ulExitCount == 2) ? (CharToUint64(chExit2) * SWAR_LANE_ONES) : m_ullStarExitBroadcast1;
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Allocates m_pszLiteralRuns and configures SWAR Optimization 3 (Shared Root Literal
    // Prefix Fast-Forward) for unbranched root literal chains of length >= CHARS_PER_QUAD,
    // updating initial states in-place to the post-prefix closure.
    // ---------------------------------------------------------------------------------
    VOID ConfigureRootLiteralPrefix(_In_reads_(ulTrieNodeCount) const TRIE_NODE* __restrict pTrieNodes,
                                    _In_                        ULONG                       ulTrieNodeCount) noexcept
    {
        ULONG ulInstructionCount = ulTrieNodeCount - 1;
        TChar* __restrict pszLiteralRuns = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                                               static_cast<SIZE_T>(ulInstructionCount) * sizeof(TChar),
                                                                               NFA_POOL_TAG));
        m_pszLiteralRuns = pszLiteralRuns;
        if (pszLiteralRuns == NULL)
        {
            return;
        }

        // Step 5: Configure SWAR Optimization 3 (Shared Root Literal Prefix Fast-Forward).
        // Evaluated before building mid-Trie literal runs so that root prefix states
        // stored at the front of m_pszLiteralRuns are marked visited and not duplicated.
        ULONG ulRootChild    = pTrieNodes[0].ulFirstChild;
        ULONG ulPrefixLen    = 0;
        ULONG ulLastPrefNode = 0;

        if (pTrieNodes[0].ulFirstMatchIdx == MAXULONG &&
            ulRootChild != MAXULONG &&
            pTrieNodes[ulRootChild].ulNextSibling == MAXULONG &&
            pTrieNodes[ulRootChild].eOp == Op::Literal)
        {
            ulPrefixLen = WalkUnbranchedLiteralChain(pTrieNodes,
                                                     ulRootChild,
                                                     &ulLastPrefNode);
        }

        if (ulPrefixLen >= CHARS_PER_QUAD)
        {
            NFA_STATE_METADATA* __restrict pStateMetadata = m_pStateMetadata;
            ULONG                          ulWalk         = 0;

            for (ULONG idx = 0; idx < ulPrefixLen; idx++)
            {
                ulWalk = pTrieNodes[ulWalk].ulFirstChild;
                pszLiteralRuns[idx] = pTrieNodes[ulWalk].chLiteral;

                // Mark root prefix state as processed (length 1) so the mid-Trie
                // literal-run pass below does not duplicate it in m_pszLiteralRuns.
                pStateMetadata[pTrieNodes[ulWalk].ulStateIdx].ulLiteralRunLength = 1;
            }

            // Replace initial states and hierarchy masks in-place with the post-prefix closure
            ULONG64* __restrict pulInitStates    = m_pulInitialStates;
            ULONG64* __restrict pulInitWordsMask = GetInitialWordsMask();
            RtlZeroMemory(pulInitStates, static_cast<SIZE_T>(m_ulStateWords64) * sizeof(ULONG64));
            RtlZeroMemory(pulInitWordsMask, static_cast<SIZE_T>(GetActiveMaskWords64()) * sizeof(ULONG64));

            ComputeClosureAndHierarchyMasks(pTrieNodes,
                                            ulLastPrefNode,
                                            pulInitStates,
                                            pulInitWordsMask,
                                            &m_ulInitialSuperMask);

            m_ulPrefixLength = ulPrefixLen;
        }
    }

    // ---------------------------------------------------------------------------------
    // Builds unbranched Op::Literal run tables in m_pszLiteralRuns (starting after any
    // shared root prefix characters) and sets NFA_INST_LITERAL_RUN_FLAG for mid-Trie
    // SWAR fast-forwarding.
    // ---------------------------------------------------------------------------------
    VOID BuildLiteralRunTables(_In_reads_(ulTrieNodeCount) const TRIE_NODE* __restrict pTrieNodes,
                               _In_                        ULONG                       ulTrieNodeCount) noexcept
    {
        // Step 6: Build unbranched Op::Literal run tables for mid-Trie SWAR fast-forwarding
        if (m_pszLiteralRuns != NULL)
        {
            TChar* __restrict              pszLiteralRuns     = m_pszLiteralRuns;
            NFA_INSTRUCTION* __restrict    pInstructions      = m_pInstructions;
            NFA_STATE_METADATA* __restrict pStateMetadata     = m_pStateMetadata;
            ULONG                          ulInstructionCount = ulTrieNodeCount - 1;
            ULONG                          ulRunCharCursor    = m_ulPrefixLength;

            for (ULONG ulNodeIdx = 1; ulNodeIdx < ulTrieNodeCount; ulNodeIdx++)
            {
                if (pTrieNodes[ulNodeIdx].eOp != Op::Literal)
                {
                    continue;
                }

                // Skip literal states already visited as part of an earlier literal chain or root prefix
                ULONG ulStateIdx = pTrieNodes[ulNodeIdx].ulStateIdx;
                if (pStateMetadata[ulStateIdx].ulLiteralRunLength != 0)
                {
                    continue;
                }

                // Walk the maximal unbranched Op::Literal chain starting at ulNodeIdx
                ULONG ulCurRunNode = ulNodeIdx;
                ULONG ulRunLen     = WalkUnbranchedLiteralChain(pTrieNodes,
                                                                ulNodeIdx,
                                                                &ulCurRunNode);

                ULONG ulEndStateIdx = pTrieNodes[ulCurRunNode].ulStateIdx;

                // Record suffix slice metadata for every state in the chain that has >= CHARS_PER_QUAD chars remaining
                if (ulRunLen >= CHARS_PER_QUAD &&
                    ulRunCharCursor + ulRunLen <= ulInstructionCount)
                {
                    ULONG ulBaseOffset = ulRunCharCursor;
                    ULONG ulWalkNode   = ulNodeIdx;

                    for (ULONG k = 0; k < ulRunLen; k++)
                    {
                        pszLiteralRuns[ulRunCharCursor++] = pTrieNodes[ulWalkNode].chLiteral;

                        ULONG               ulWalkState = pTrieNodes[ulWalkNode].ulStateIdx;
                        NFA_STATE_METADATA& rWalkMeta   = pStateMetadata[ulWalkState];
                        ULONG               ulRemLen    = ulRunLen - k;

                        rWalkMeta.ulLiteralRunOffset   = ulBaseOffset + k;
                        rWalkMeta.ulLiteralRunLength   = ulRemLen;
                        rWalkMeta.ulLiteralRunEndState = ulEndStateIdx;

                        // Flag hot instruction so VM can detect SWAR literal runs without touching cold metadata
                        if (ulRemLen >= CHARS_PER_QUAD)
                        {
                            pInstructions[ulWalkState].ulExtraOffset |= NFA_INST_LITERAL_RUN_FLAG;
                        }

                        ulWalkNode = pTrieNodes[ulWalkNode].ulFirstChild;
                    }
                }
                else
                {
                    // Mark short literal chains (< CHARS_PER_QUAD) as visited so their sub-chains aren't rescanned
                    ULONG ulWalkNode = ulNodeIdx;
                    for (ULONG k = 0; k < ulRunLen; k++)
                    {
                        ULONG ulWalkState = pTrieNodes[ulWalkNode].ulStateIdx;
                        pStateMetadata[ulWalkState].ulLiteralRunLength = 1;
                        ulWalkNode = pTrieNodes[ulWalkNode].ulFirstChild;
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 3A: Compares an unbranched run of Op::Literal characters
    // (>= CHARS_PER_QUAD) 8 bytes (4 WCHARs or 8 CHARs) at a time in-register, using
    // an overlapping 64-bit tail load at (ulRunLen - CHARS_PER_QUAD) for unaligned tails.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline BOOLEAN VerifyLiteralRunSwar(_In_reads_(ulRunLen) const TChar* __restrict pszText,
                                               _In_reads_(ulRunLen) const TChar* __restrict pszExpected,
                                               _In_                 ULONG                   ulRunLen) const noexcept
    {
        // Compare full 64-bit quads (CHARS_PER_QUAD characters per step)
        for (ULONG i = 0; i + CHARS_PER_QUAD <= ulRunLen; i += CHARS_PER_QUAD)
        {
            if (ReadNormalizedQuad<bCaseInsensitive>(pszText + i) != Read64Safe(pszExpected + i))
            {
                return FALSE;
            }
        }

        // Handle remaining unaligned tail characters via an overlapping 64-bit load
        if ((ulRunLen & CHARS_PER_QUAD_MASK) != 0)
        {
            ULONG ulTailIdx = ulRunLen - CHARS_PER_QUAD;
            if (ReadNormalizedQuad<bCaseInsensitive>(pszText + ulTailIdx) != Read64Safe(pszExpected + ulTailIdx))
            {
                return FALSE;
            }
        }

        return TRUE;
    }

    // ---------------------------------------------------------------------------------
    // SWAR Optimization 3B: Verifies a shared anchored literal prefix 8 bytes (4 WCHARs
    // or 8 CHARs) at a time before entering the NFA state machine.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline BOOLEAN VerifyRootLiteralPrefixSwar(_In_reads_(cchLen) const TChar* __restrict pszText,
                                                      _In_               size_t                  cchLen) const noexcept
    {
        // Input shorter than the mandatory shared root prefix cannot match any pattern
        if (cchLen < m_ulPrefixLength)
        {
            return FALSE;
        }

        return VerifyLiteralRunSwar<bCaseInsensitive>(pszText, m_pszLiteralRuns, m_ulPrefixLength);
    }

    // ---------------------------------------------------------------------------------
    // Checks whether an un-killable Word-0 WILD_CARD_SCOPE::SCOPE_DEFAULT Op::Star accepting state is
    // active during a MatchFirst query, resolving its first pattern context if so.
    // ---------------------------------------------------------------------------------
    __forceinline BOOLEAN CheckWord0PermanentAccept(_In_      ULONG64              ullStates0,
                                                    _In_      BOOLEAN              bMatchFirst,
                                                    _Out_opt_ TContext* __restrict pOutContexts) const noexcept
    {
        // Intersect active Word-0 states with precomputed permanent '*' accepting mask
        ULONG64 ullPermAcceptBits = ullStates0 & m_ullWord0PermanentAcceptMask;
        if (!bMatchFirst || ullPermAcceptBits == 0)
        {
            return FALSE;
        }

        // Resolve the lowest-numbered active permanent accepting state's first context
        if (pOutContexts != NULL)
        {
            const NFA_STATE_METADATA* __restrict pStateMetadata = m_pStateMetadata;
            const MATCH_LINK_NODE* __restrict    pMatchLinks    = m_pMatchLinks;
            ULONG                                ulStateBit     = 0;

            _BitScanForward64(&ulStateBit, ullPermAcceptBits);

            ULONG ulMatchNodeIdx = pStateMetadata[ulStateBit].ulFirstMatchIdx;
            *pOutContexts        = pMatchLinks[ulMatchNodeIdx].Context;
        }

        return TRUE;
    }

    // ---------------------------------------------------------------------------------
    // Attempts to fast-forward over an unbranched Op::Literal run >= CHARS_PER_QUAD
    // when only a single NFA state is currently active.
    // Checks NFA_INST_LITERAL_RUN_FLAG directly on the hot 16-byte NFA_INSTRUCTION so
    // states without literal runs never touch the cold m_pStateMetadata array.
    // Returns:
    //   0 : State does not start a literal run >= CHARS_PER_QUAD (continue normal step).
    //   1 : Run matched and *pcchOffset was advanced; *ppEndInst receives the end state.
    //  -1 : Run failed to match or reached a dead-end state (abort VM with FALSE).
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    __forceinline INT TryAdvanceLiteralRunSwar(_In_               ULONG                              ulSingleStateIdx,
                                               _In_reads_(cchLen) const TChar* __restrict            pszText,
                                               _In_               size_t                             cchLen,
                                               _Inout_            size_t* __restrict                 pcchOffset,
                                               _Out_              const NFA_INSTRUCTION** __restrict ppEndInst) const noexcept
    {
        // Fast check on the hot instruction word before touching cold state metadata
        const NFA_INSTRUCTION* __restrict pInstructions = m_pInstructions;
        const NFA_INSTRUCTION&            rSingleInst   = pInstructions[ulSingleStateIdx];
        if ((rSingleInst.ulExtraOffset & NFA_INST_LITERAL_RUN_FLAG) == 0)
        {
            *ppEndInst = NULL;
            return 0;
        }

        const NFA_STATE_METADATA& __restrict rSingleMeta = m_pStateMetadata[ulSingleStateIdx];
        ULONG                                ulRunLen    = rSingleMeta.ulLiteralRunLength;

        // Fail immediately if remaining input is shorter than the literal run or fails SWAR comparison
        size_t cchOffset = *pcchOffset;
        if (cchLen - cchOffset < ulRunLen ||
            !VerifyLiteralRunSwar<bCaseInsensitive>(pszText + cchOffset,
                                                    m_pszLiteralRuns + rSingleMeta.ulLiteralRunOffset,
                                                    ulRunLen))
        {
            *ppEndInst = NULL;
            return -1;
        }

        // Retrieve the instruction at the end of the literal chain to apply its outgoing transitions
        const NFA_INSTRUCTION& rEndInst = pInstructions[rSingleMeta.ulLiteralRunEndState];
        if (rEndInst.usTargetWord == MAXUSHORT)
        {
            *ppEndInst = NULL;
            return -1;
        }

        *pcchOffset = cchOffset + ulRunLen;
        *ppEndInst  = &rEndInst;
        return 1;
    }

    // ---------------------------------------------------------------------------------
    // Applies all outgoing target state bits (primary target word + any overflow target
    // words) of rInst into the Path A (<= 63 state words) state and mask buffers.
    // Derives the 64-bit word-mask bit in-register via (1ULL << (ulTargetWord & 63)).
    // ---------------------------------------------------------------------------------
    __forceinline VOID ApplyInstructionTargetsStack(_In_                              const NFA_INSTRUCTION& __restrict rInst,
                                                    _Inout_updates_(m_ulStateWords64) ULONG64* __restrict               pTargetStates,
                                                    _Inout_                           ULONG64& __restrict               rulWordsMask) const noexcept
    {
        // Apply primary inline target word and set its corresponding bit in rulWordsMask
        ULONG ulTargetWord          = rInst.usTargetWord;
        pTargetStates[ulTargetWord] |= rInst.ullTargetBits;
        rulWordsMask                |= (1ULL << (ulTargetWord & 63));

        // Apply any overflow target words (masking off NFA_INST_LITERAL_RUN_FLAG defensively)
        if (rInst.usExtraCount != 0)
        {
            const EXTRA_TARGET_WORD* __restrict pExtra = &m_pExtraTargets[rInst.ulExtraOffset & ~NFA_INST_LITERAL_RUN_FLAG];
            for (ULONG e = 0; e < rInst.usExtraCount; e++)
            {
                ULONG ulExtraWord          = pExtra[e].ulTargetWord;
                pTargetStates[ulExtraWord] |= pExtra[e].ullTargetBits;
                rulWordsMask               |= (1ULL << (ulExtraWord & 63));
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Applies all outgoing target state bits (primary target word + any overflow target
    // words) of rInst into the Path B (> 63 state words) 3-level hierarchy buffers.
    // Derives 2-level and 3-level hierarchy mask bits in-register from ulTargetWord.
    // ---------------------------------------------------------------------------------
    __forceinline VOID ApplyInstructionTargetsSuper(_In_                              const NFA_INSTRUCTION& __restrict rInst,
                                                    _Inout_updates_(m_ulStateWords64) ULONG64* __restrict               pTargetStates,
                                                    _Inout_                           ULONG64* __restrict               pTargetWordsMask,
                                                    _Inout_                           ULONG64& __restrict               rulSuperMask) const noexcept
    {
        // Apply primary target word and propagate dirty bits up to 2nd-level mask and 3rd-level supermask
        ULONG ulTargetWord          = rInst.usTargetWord;
        ULONG ulMaskIdx             = ulTargetWord >> 6;

        pTargetStates[ulTargetWord] |= rInst.ullTargetBits;
        pTargetWordsMask[ulMaskIdx] |= (1ULL << (ulTargetWord & 63));
        rulSuperMask                |= (1ULL << (ulMaskIdx & 63));

        // Apply any overflow target words across the 3-level bitset hierarchy
        if (rInst.usExtraCount != 0)
        {
            const EXTRA_TARGET_WORD* __restrict pExtra = &m_pExtraTargets[rInst.ulExtraOffset & ~NFA_INST_LITERAL_RUN_FLAG];
            for (ULONG e = 0; e < rInst.usExtraCount; e++)
            {
                ULONG ulExtraWord            = pExtra[e].ulTargetWord;
                ULONG ulExtraMask            = ulExtraWord >> 6;

                pTargetStates[ulExtraWord]   |= pExtra[e].ullTargetBits;
                pTargetWordsMask[ulExtraMask] |= (1ULL << (ulExtraWord & 63));
                rulSuperMask                 |= (1ULL << (ulExtraMask & 63));
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Extracts all matching pattern contexts from a surviving state word at the end of
    // input evaluation. Returns TRUE if bMatchFirst is set and a match was resolved.
    // ---------------------------------------------------------------------------------
    __forceinline BOOLEAN HarvestWordMatches(_In_                           ULONG                ulWordIdx,
                                             _In_                           ULONG64              ullWordStates,
                                             _Out_writes_opt_(ulMaxMatches) TContext* __restrict pOutContexts,
                                             _In_                           ULONG                ulMaxMatches,
                                             _Inout_                        ULONG& __restrict    rulMatched,
                                             _In_                           BOOLEAN              bMatchFirst) const noexcept
    {
        // Filter surviving active states in this word down to accepting states only
        const ULONG64* __restrict            pulAcceptStates = GetAcceptStates();
        const NFA_STATE_METADATA* __restrict pStateMetadata  = m_pStateMetadata;
        const MATCH_LINK_NODE* __restrict    pMatchLinks     = m_pMatchLinks;
        ULONG64                              ulAcceptBits    = ullWordStates & pulAcceptStates[ulWordIdx];
        ULONG                                ulBaseStateIdx  = ulWordIdx << 6;

        while (ulAcceptBits != 0)
        {
            ULONG ulStateBit = 0;
            _BitScanForward64(&ulStateBit, ulAcceptBits);
            ulAcceptBits &= (ulAcceptBits - 1);

            // Walk the singly-linked list of pattern contexts associated with this accepting state
            ULONG ulMatchNodeIdx = pStateMetadata[ulBaseStateIdx + ulStateBit].ulFirstMatchIdx;

            while (ulMatchNodeIdx != MAXULONG)
            {
                const MATCH_LINK_NODE& rMatchLink = pMatchLinks[ulMatchNodeIdx];

                if (bMatchFirst)
                {
                    if (pOutContexts != NULL)
                    {
                        *pOutContexts = rMatchLink.Context;
                    }

                    return TRUE;
                }

                if (pOutContexts != NULL && rulMatched < ulMaxMatches)
                {
                    pOutContexts[rulMatched] = rMatchLink.Context;
                }

                rulMatched++;
                ulMatchNodeIdx = rMatchLink.ulNextMatchIdx;
            }
        }

        return FALSE;
    }

    // ---------------------------------------------------------------------------------
    // Core VM execution engine shared by MatchFirst and MatchAll.
    // Integrates register-direct 64-bit SWAR scanning, dynamic Word-0 steady-state
    // skipping, permanent accept early-exit, unbranched literal-run fast-forwarding,
    // and lock-free sparse scratch buffer management.
    // ---------------------------------------------------------------------------------
    template <BOOLEAN bCaseInsensitive>
    _IRQL_requires_max_(DISPATCH_LEVEL)
    BOOLEAN ExecuteVm(_In_reads_or_z_(cchLen)        const TChar* __restrict pszText,
                      _In_                           size_t                  cchLen,
                      _Out_writes_opt_(ulMaxMatches) TContext* __restrict    pOutContexts,
                      _In_                           ULONG                   ulMaxMatches,
                      _Out_opt_                      ULONG* __restrict       pulMatchCount,
                      _In_                           BOOLEAN                 bMatchFirst) const noexcept
    {
        if (pulMatchCount != NULL)
        {
            *pulMatchCount = 0;
        }

        // Pair with WriteBooleanRelease in Compile() to ensure all NFA tables are visible
        if (!ReadBooleanAcquire(&m_bCompiled))
        {
            return FALSE;
        }

        if (pszText == NULL && cchLen > 0)
        {
            return FALSE;
        }

        // SWAR Optimization 3: Fast-forward over shared anchored literal prefix (if any).
        // When m_ulPrefixLength > 0, m_pulInitialStates, GetInitialWordsMask(), and
        // m_ulInitialSuperMask already hold the precomputed post-prefix closure.
        if (m_ulPrefixLength > 0 && !VerifyRootLiteralPrefixSwar<bCaseInsensitive>(pszText, cchLen))
        {
            return FALSE;
        }

        size_t                            cchOffset     = m_ulPrefixLength;
        const ULONG64* __restrict         pulInitStates = m_pulInitialStates;
        const ULONG64* __restrict         pulInitMask   = GetInitialWordsMask();
        const NFA_INSTRUCTION* __restrict pInstructions = m_pInstructions;
        ULONG64                           ulInitSuper   = m_ulInitialSuperMask;

        // -----------------------------------------------------------------------------
        // PATH A: <= 63 State Words Stack Fast-Path
        // Maintains all active states on the stack and tracks non-zero state words
        // inside a single 64-bit register (ulCurrentWordsMask).
        // -----------------------------------------------------------------------------
        if (m_ulStateWords64 <= STACK_FAST_PATH_WORDS)
        {
            ULONG64  StatesBufferA[STACK_FAST_PATH_WORDS];
            ULONG64  StatesBufferB[STACK_FAST_PATH_WORDS];

            ULONG64* pCurrentStates     = StatesBufferA;
            ULONG64* pNextStates        = StatesBufferB;
            SIZE_T   cbStateBytes       = static_cast<SIZE_T>(m_ulStateWords64) * sizeof(ULONG64);

            RtlCopyMemory(pCurrentStates, pulInitStates, cbStateBytes);
            RtlZeroMemory(pNextStates, cbStateBytes);

            ULONG64  ulCurrentWordsMask = pulInitMask[0];
            ULONG64  ulNextWordsMask    = 0;

            if (ulCurrentWordsMask == 0)
            {
                return FALSE;
            }

            while (cchOffset < cchLen)
            {
                // Single consolidated early-exit check covering initial states,
                // post-literal-run states, and per-character transition states.
                if (CheckWord0PermanentAccept(pCurrentStates[0], bMatchFirst, pOutContexts))
                {
                    return TRUE;
                }

                // SWAR Optimization 2: Root & Dynamic Word-0 '*' Steady-State 64-Bit Skip
                if (ulCurrentWordsMask == 1ULL)
                {
                    TrySkipWord0SteadyStateSwar<bCaseInsensitive>(pCurrentStates[0],
                                                                  pszText,
                                                                  cchLen,
                                                                  &cchOffset);

                    if (cchOffset >= cchLen)
                    {
                        break;
                    }
                }

                // SWAR Optimization 3A: Unbranched Literal-Run Fast-Forward when 1 state is active.                
                if ((ulCurrentWordsMask & (ulCurrentWordsMask - 1)) == 0)
                {
                    ULONG ulSingleWordIdx = 0;
                    _BitScanForward64(&ulSingleWordIdx, ulCurrentWordsMask);

                    ULONG64 ullSingleWordStates = pCurrentStates[ulSingleWordIdx];
                    if ((ullSingleWordStates & (ullSingleWordStates - 1)) == 0)
                    {
                        ULONG ulSingleBit = 0;
                        _BitScanForward64(&ulSingleBit, ullSingleWordStates);

                        const NFA_INSTRUCTION* pEndInst   = NULL;
                        INT                    nRunStatus = TryAdvanceLiteralRunSwar<bCaseInsensitive>((ulSingleWordIdx << 6) + ulSingleBit,
                                                                                                       pszText,
                                                                                                       cchLen,
                                                                                                       &cchOffset,
                                                                                                       &pEndInst);

                        if (nRunStatus < 0)
                        {
                            return FALSE;
                        }

                        if (nRunStatus > 0)
                        {
                            // Clear the single starting state and apply the end-of-run state's outgoing transitions in-place
                            pCurrentStates[ulSingleWordIdx] = 0;
                            ulCurrentWordsMask              = 0;

                            ApplyInstructionTargetsStack(*pEndInst,
                                                         pCurrentStates,
                                                         ulCurrentWordsMask);
                            continue;
                        }
                    }
                }

                // ASCII characters (0x00..0x7F) are already case-folded inside ullAsciiMatchMask[128];
                // only non-ASCII 16-bit WCHARs (>= 0x80) require runtime UpcaseChar() translation.
                TChar ch = pszText[cchOffset++];
                if constexpr (bCaseInsensitive && sizeof(TChar) == 2)
                {
                    if (static_cast<USHORT>(ch) >= 0x80) [[unlikely]]
                    {
                        ch = UpcaseChar(ch);
                    }
                }

                // Scoped restricted aliases valid strictly for this character step prior to buffer swap
                {
                    ULONG64* __restrict pCurStates   = pCurrentStates;
                    ULONG64* __restrict pNxtStates   = pNextStates;
                    ULONG64             ulActiveMask = ulCurrentWordsMask;

                    // Iterate only over non-zero 64-bit state words using Kernighan bit-clearing
                    while (ulActiveMask != 0)
                    {
                        ULONG ulWordIdx = 0;
                        _BitScanForward64(&ulWordIdx, ulActiveMask);
                        ulActiveMask &= (ulActiveMask - 1);

                        // Consume and zero pCurStates[ulWordIdx] inline so it is clean when buffers swap
                        ULONG64 ulRawStates   = pCurStates[ulWordIdx];
                        pCurStates[ulWordIdx] = 0;

                        // SWAR Optimization 4: Filter all 64 states in parallel before bit-scan
                        ULONG64 ulActiveStates = FilterActiveStatesSwar(ulWordIdx, ulRawStates, ch);
                        ULONG   ulBaseStateIdx = ulWordIdx << 6;

                        while (ulActiveStates != 0)
                        {
                            ULONG ulStateBit = 0;
                            _BitScanForward64(&ulStateBit, ulActiveStates);
                            ulActiveStates &= (ulActiveStates - 1);

                            ApplyInstructionTargetsStack(pInstructions[ulBaseStateIdx + ulStateBit],
                                                         pNxtStates,
                                                         ulNextWordsMask);
                        }
                    }
                }

                // Single-instruction early exit if all active states died out
                if (!ulNextWordsMask)
                {
                    return FALSE;
                }

                // Swap current and next state buffers for the next input character
                ULONG64* pTempStates = pCurrentStates;
                pCurrentStates       = pNextStates;
                pNextStates          = pTempStates;

                ulCurrentWordsMask   = ulNextWordsMask;
                ulNextWordsMask      = 0;
            }

            // Extract matched pattern contexts from surviving accepting states
            ULONG ulMatched = 0;

            while (ulCurrentWordsMask != 0)
            {
                ULONG ulWordIdx = 0;
                _BitScanForward64(&ulWordIdx, ulCurrentWordsMask);
                ulCurrentWordsMask &= (ulCurrentWordsMask - 1);

                if (HarvestWordMatches(ulWordIdx,
                                       pCurrentStates[ulWordIdx],
                                       pOutContexts,
                                       ulMaxMatches,
                                       ulMatched,
                                       bMatchFirst))
                {
                    return TRUE;
                }
            }

            if (pulMatchCount != NULL)
            {
                *pulMatchCount = ulMatched;
            }

            return (ulMatched > 0);
        }

        // -----------------------------------------------------------------------------
        // PATH B: > 63 State Words Hierarchical ulSuperMask Path
        // Uses a 64-bit register (ulSuperMask) where bit m indicates that
        // pCurrentWordsMask[m] is non-zero. Uses a lock-free pre-zeroed member scratch
        // buffer with sparse ulSuperMask initialization and cleanup.
        // -----------------------------------------------------------------------------
        else
        {
            if (!ulInitSuper)
            {
                return FALSE;
            }

            ULONG               ulActiveMaskWords64 = GetActiveMaskWords64();
            SIZE_T              cbScratchWords      = static_cast<SIZE_T>(m_ulStateWords64 + ulActiveMaskWords64) * 2;
            ULONG64* __restrict pScratchBase        = NULL;
            BOOLEAN             bUsingMemberBuffer  = FALSE;

            // Attempt lock-free acquisition of the pre-zeroed member scratch buffer;
            // fall back to a per-call non-paged pool allocation under concurrency.
            // Uses Acquire semantics to prevent ARM64 from hoisting scratch buffer reads/writes.
            if (m_pulScratchStateBuffer != NULL &&
                InterlockedCompareExchangeAcquire(&m_lScratchBufferInUse, 1, 0) == 0)
            {
                pScratchBase       = m_pulScratchStateBuffer;
                bUsingMemberBuffer = TRUE;
            }
            else
            {
                pScratchBase = static_cast<ULONG64*>(ExAllocatePool2(POOL_FLAG_NON_PAGED,
                                                                     cbScratchWords * sizeof(ULONG64),
                                                                     NFA_POOL_TAG));

                if (pScratchBase == NULL)
                {
                    return FALSE;
                }
            }

            // Partition the scratch allocation into double-buffered state and 2nd-level mask arrays
            ULONG64* pCurrentStates    = pScratchBase;
            ULONG64* pNextStates       = pScratchBase + m_ulStateWords64;
            ULONG64* pCurrentWordsMask = pNextStates + m_ulStateWords64;
            ULONG64* pNextWordsMask    = pCurrentWordsMask + ulActiveMaskWords64;

            ULONG64  ulSuperMask       = ulInitSuper;
            ULONG64  ulNextSuperMask   = 0;

            // Sparse ulSuperMask initialization: touch only active mask and state words
            ULONG64  ulInitSuperWalk = ulInitSuper;
            while (ulInitSuperWalk != 0)
            {
                ULONG ulM = 0;
                _BitScanForward64(&ulM, ulInitSuperWalk);
                ulInitSuperWalk &= (ulInitSuperWalk - 1);

                ULONG64 ulWordsMask    = pulInitMask[ulM];
                pCurrentWordsMask[ulM] = ulWordsMask;
                ULONG   ulBaseWordIdx  = ulM << 6;

                while (ulWordsMask != 0)
                {
                    ULONG ulWordBit = 0;
                    _BitScanForward64(&ulWordBit, ulWordsMask);
                    ulWordsMask &= (ulWordsMask - 1);

                    ULONG ulWordIdx           = ulBaseWordIdx + ulWordBit;
                    pCurrentStates[ulWordIdx] = pulInitStates[ulWordIdx];
                }
            }

            // Helper lambda performing sparse ulSuperMask cleanup before releasing scratch buffer
            // so the member buffer remains strictly pre-zeroed for the next caller.
            // Uses WriteRelease so all zeroing writes are globally visible on ARM64/x64
            // before m_lScratchBufferInUse is cleared.
            auto ReleaseScratchBuffer = [&]() noexcept -> VOID
            {
                if (bUsingMemberBuffer)
                {
                    while (ulSuperMask != 0)
                    {
                        ULONG ulM = 0;
                        _BitScanForward64(&ulM, ulSuperMask);
                        ulSuperMask &= (ulSuperMask - 1);

                        ULONG64 ulWordsMask    = pCurrentWordsMask[ulM];
                        pCurrentWordsMask[ulM] = 0;
                        ULONG   ulBaseWordIdx  = ulM << 6;

                        while (ulWordsMask != 0)
                        {
                            ULONG ulWordBit = 0;
                            _BitScanForward64(&ulWordBit, ulWordsMask);
                            ulWordsMask &= (ulWordsMask - 1);

                            pCurrentStates[ulBaseWordIdx + ulWordBit] = 0;
                        }
                    }

                    WriteRelease(&m_lScratchBufferInUse, 0);
                }
                else
                {
                    ExFreePoolWithTag(pScratchBase, NFA_POOL_TAG);
                }
            };

            while (cchOffset < cchLen)
            {
                // Single consolidated early-exit check covering initial states,
                // post-literal-run states, and per-character transition states.
                if (CheckWord0PermanentAccept(pCurrentStates[0], bMatchFirst, pOutContexts))
                {
                    ReleaseScratchBuffer();
                    return TRUE;
                }

                // SWAR Optimization 2: Root & Dynamic Word-0 '*' Steady-State 64-Bit Skip
                if (ulSuperMask == 1ULL && pCurrentWordsMask[0] == 1ULL)
                {
                    TrySkipWord0SteadyStateSwar<bCaseInsensitive>(pCurrentStates[0],
                                                                  pszText,
                                                                  cchLen,
                                                                  &cchOffset);
                    if (cchOffset >= cchLen)
                    {
                        break;
                    }
                }

                // SWAR Optimization 3A: Unbranched Literal-Run Fast-Forward when 1 state is active                
                if ((ulSuperMask & (ulSuperMask - 1)) == 0)
                {
                    ULONG ulSingleM = 0;
                    _BitScanForward64(&ulSingleM, ulSuperMask);

                    ULONG64 ullSingleWordsMask = pCurrentWordsMask[ulSingleM];
                    if ((ullSingleWordsMask & (ullSingleWordsMask - 1)) == 0)
                    {
                        ULONG ulSingleWordBit = 0;
                        _BitScanForward64(&ulSingleWordBit, ullSingleWordsMask);

                        ULONG   ulSingleWordIdx     = (ulSingleM << 6) + ulSingleWordBit;
                        ULONG64 ullSingleWordStates = pCurrentStates[ulSingleWordIdx];

                        if ((ullSingleWordStates & (ullSingleWordStates - 1)) == 0)
                        {
                            ULONG ulSingleBit = 0;
                            _BitScanForward64(&ulSingleBit, ullSingleWordStates);

                            const NFA_INSTRUCTION* pEndInst   = NULL;
                            INT                    nRunStatus = TryAdvanceLiteralRunSwar<bCaseInsensitive>((ulSingleWordIdx << 6) + ulSingleBit,
                                                                                                           pszText,
                                                                                                           cchLen,
                                                                                                           &cchOffset,
                                                                                                           &pEndInst);

                            if (nRunStatus < 0)
                            {
                                ReleaseScratchBuffer();
                                return FALSE;
                            }

                            if (nRunStatus > 0)
                            {
                                // Clear the single active state across all 3 hierarchy levels and apply end-of-run targets
                                pCurrentStates[ulSingleWordIdx] = 0;
                                pCurrentWordsMask[ulSingleM]    = 0;
                                ulSuperMask                     = 0;

                                ApplyInstructionTargetsSuper(*pEndInst,
                                                             pCurrentStates,
                                                             pCurrentWordsMask,
                                                             ulSuperMask);
                                continue;
                            }
                        }
                    }
                }

                // ASCII characters (0x00..0x7F) are already case-folded inside ullAsciiMatchMask[128];
                // only non-ASCII 16-bit WCHARs (>= 0x80) require runtime UpcaseChar() translation.
                TChar ch = pszText[cchOffset++];
                if constexpr (bCaseInsensitive && sizeof(TChar) == 2)
                {
                    if (static_cast<USHORT>(ch) >= 0x80) [[unlikely]]
                    {
                        ch = UpcaseChar(ch);
                    }
                }

                // Scoped restricted aliases valid strictly for this character step prior to buffer swap
                {
                    ULONG64* __restrict pCurStates    = pCurrentStates;
                    ULONG64* __restrict pNxtStates    = pNextStates;
                    ULONG64* __restrict pCurWordsMask = pCurrentWordsMask;
                    ULONG64* __restrict pNxtWordsMask = pNextWordsMask;
                    ULONG64             ulActiveSuper = ulSuperMask;

                    // Traverse the 3-level hierarchy (SuperMask -> WordsMask -> StateWord), zeroing consumed words inline
                    while (ulActiveSuper != 0)
                    {
                        ULONG ulM = 0;
                        _BitScanForward64(&ulM, ulActiveSuper);
                        ulActiveSuper &= (ulActiveSuper - 1);

                        ULONG64 ulWordsMask   = pCurWordsMask[ulM];
                        pCurWordsMask[ulM]    = 0;
                        ULONG   ulBaseWordIdx = ulM << 6;

                        while (ulWordsMask != 0)
                        {
                            ULONG ulWordBit = 0;
                            _BitScanForward64(&ulWordBit, ulWordsMask);
                            ulWordsMask &= (ulWordsMask - 1);

                            ULONG   ulWordIdx      = ulBaseWordIdx + ulWordBit;
                            ULONG64 ulRawStates    = pCurStates[ulWordIdx];
                            pCurStates[ulWordIdx]  = 0;

                            // SWAR Optimization 4: Filter all 64 states in parallel before bit-scan
                            ULONG64 ulActiveStates = FilterActiveStatesSwar(ulWordIdx, ulRawStates, ch);
                            ULONG   ulBaseStateIdx = ulWordIdx << 6;

                            while (ulActiveStates != 0)
                            {
                                ULONG ulStateBit = 0;
                                _BitScanForward64(&ulStateBit, ulActiveStates);
                                ulActiveStates &= (ulActiveStates - 1);

                                ApplyInstructionTargetsSuper(pInstructions[ulBaseStateIdx + ulStateBit],
                                                             pNxtStates,
                                                             pNxtWordsMask,
                                                             ulNextSuperMask);
                            }
                        }
                    }
                }

                // Single-instruction early exit check when all states die out
                // Note: pCurrentStates and pCurrentWordsMask were zeroed inline above
                if (!ulNextSuperMask)
                {
                    ulSuperMask = 0;
                    ReleaseScratchBuffer();
                    return FALSE;
                }

                // Swap current and next state and 2nd-level mask buffers for the next character step
                ULONG64* pTempStates = pCurrentStates;
                pCurrentStates       = pNextStates;
                pNextStates          = pTempStates;

                ULONG64* pTempMask   = pCurrentWordsMask;
                pCurrentWordsMask    = pNextWordsMask;
                pNextWordsMask       = pTempMask;

                ulSuperMask          = ulNextSuperMask;
                ulNextSuperMask      = 0;
            }

            // Extract matched pattern contexts using a copy of ulSuperMask so
            // ReleaseScratchBuffer() can perform sparse cleanup uniformly
            ULONG   ulMatched     = 0;
            ULONG64 ulAcceptSuper = ulSuperMask;

            while (ulAcceptSuper != 0)
            {
                ULONG ulM = 0;
                _BitScanForward64(&ulM, ulAcceptSuper);
                ulAcceptSuper &= (ulAcceptSuper - 1);

                ULONG64 ulWordsMask   = pCurrentWordsMask[ulM];
                ULONG   ulBaseWordIdx = ulM << 6;

                while (ulWordsMask != 0)
                {
                    ULONG ulWordBit = 0;
                    _BitScanForward64(&ulWordBit, ulWordsMask);
                    ulWordsMask &= (ulWordsMask - 1);

                    ULONG ulWordIdx = ulBaseWordIdx + ulWordBit;

                    if (HarvestWordMatches(ulWordIdx,
                                           pCurrentStates[ulWordIdx],
                                           pOutContexts,
                                           ulMaxMatches,
                                           ulMatched,
                                           bMatchFirst))
                    {
                        ReleaseScratchBuffer();
                        return TRUE;
                    }
                }
            }

            ReleaseScratchBuffer();

            if (pulMatchCount != NULL)
            {
                *pulMatchCount = ulMatched;
            }

            return (ulMatched > 0);
        }
    }

public:

    // ---------------------------------------------------------------------------------
    // Constructs an empty NFA instance configured for case-insensitive or exact mode.
    // ---------------------------------------------------------------------------------
    explicit KmWildcardNfa(_In_ BOOLEAN bCaseInsensitive = TRUE ) noexcept : m_bCaseInsensitive(bCaseInsensitive),
                                                                             m_bCompiled(FALSE),
                                                                             m_ulStateWords64(0),
                                                                             m_ulPrefixLength(0),
                                                                             m_pInstructions(NULL),
                                                                             m_pStateMetadata(NULL),
                                                                             m_pWordFilters(NULL),
                                                                             m_pExtraTargets(NULL),
                                                                             m_pMatchLinks(NULL),
                                                                             m_pulInitialStates(NULL),
                                                                             m_pszLiteralRuns(NULL),
                                                                             m_ulInitialSuperMask(0),
                                                                             m_ullSteadyStateWord0(0),
                                                                             m_ullStarExitBroadcast1(0),
                                                                             m_ullStarExitBroadcast2(0),
                                                                             m_ullWord0DefaultStarMask(0),
                                                                             m_ullWord0PermanentAcceptMask(0),
                                                                             m_pulScratchStateBuffer(NULL),
                                                                             m_pPatterns(NULL),
                                                                             m_ulPatternCount(0),
                                                                             m_ulPatternCapacity(0),
                                                                             m_lScratchBufferInUse(0)
    {
    }

    // ---------------------------------------------------------------------------------
    // Destructor freeing all pattern allocations and compiled NFA state tables.
    // ---------------------------------------------------------------------------------
    ~KmWildcardNfa() noexcept
    {
        Clear();
    }

    KmWildcardNfa(const KmWildcardNfa&) = delete;
    KmWildcardNfa& operator=(const KmWildcardNfa&) = delete;

    // ---------------------------------------------------------------------------------
    // Clears all registered patterns and frees compiled state, returning the engine
    // to a clean slate without requiring destruction and reallocation.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(DISPATCH_LEVEL)
    VOID Clear() noexcept
    {
        FreeCompiledBuffers();

        // Free each pattern's individual character buffer followed by the staging array
        if (m_pPatterns != NULL)
        {
            for (ULONG i = 0; i < m_ulPatternCount; i++)
            {
                if (m_pPatterns[i].pszPattern != NULL)
                {
                    ExFreePoolWithTag(m_pPatterns[i].pszPattern, NFA_POOL_TAG);
                }
            }

            ExFreePoolWithTag(m_pPatterns, NFA_POOL_TAG);
            m_pPatterns = NULL;
        }

        m_ulPatternCount    = 0;
        m_ulPatternCapacity = 0;
    }

    // ---------------------------------------------------------------------------------
    // Adds a wildcard pattern and associated context to the staging array.
    // Pre-normalizes pattern casing at insertion time via 64-bit SWAR if
    // m_bCaseInsensitive is enabled.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(APC_LEVEL)
    NTSTATUS AddPattern(_In_reads_or_z_(cchLen) const TChar* __restrict    pszPattern,
                        _In_                    size_t                     cchLen,
                        _In_                    WILD_CARD_SCOPE            eScope,
                        _In_                    const TContext& __restrict rContext) noexcept
    {
        // Validate pointer, byte-size overflow limits, and wildcard scope enumerator
        if ((pszPattern == NULL && cchLen > 0) ||
            cchLen > (MAXULONG / sizeof(TChar)) ||
            (eScope != WILD_CARD_SCOPE::SCOPE_DEFAULT && eScope != WILD_CARD_SCOPE::SCOPE_PATH_SEGMENT))
        {
            return STATUS_INVALID_PARAMETER;
        }

        // Adding a new pattern invalidates any previously compiled NFA tables
        if (ReadBooleanAcquire(&m_bCompiled))
        {
            FreeCompiledBuffers();
        }

        // Grow the pattern staging array geometrically (starting at 16 entries) when full
        if (m_ulPatternCount == m_ulPatternCapacity)
        {
            if (m_ulPatternCapacity > (MAXULONG / 2))
            {
                return STATUS_INSUFFICIENT_RESOURCES;
            }

            ULONG                     ulNewCapacity = (m_ulPatternCapacity == 0) ? 16 : (m_ulPatternCapacity * 2);
            SIZE_T                    cbAllocSize   = static_cast<SIZE_T>(ulNewCapacity) * sizeof(PATTERN_ENTRY);
            PATTERN_ENTRY* __restrict pNewPatterns  = static_cast<PATTERN_ENTRY*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                                                                  cbAllocSize,
                                                                                                  NFA_POOL_TAG));
            if (pNewPatterns == NULL)
            {
                return STATUS_INSUFFICIENT_RESOURCES;
            }

            if (m_pPatterns != NULL)
            {
                RtlCopyMemory(pNewPatterns,
                              m_pPatterns,
                              static_cast<SIZE_T>(m_ulPatternCount) * sizeof(PATTERN_ENTRY));
                ExFreePoolWithTag(m_pPatterns, NFA_POOL_TAG);
            }

            m_pPatterns         = pNewPatterns;
            m_ulPatternCapacity = ulNewCapacity;
        }

        TChar* __restrict pszCopy = NULL;

        // Allocate and copy (or SWAR case-fold) the pattern string into non-paged pool
        if (cchLen > 0)
        {
            pszCopy = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_NON_PAGED | POOL_FLAG_UNINITIALIZED,
                                                          cchLen * sizeof(TChar),
                                                          NFA_POOL_TAG));
            if (pszCopy == NULL)
            {
                return STATUS_INSUFFICIENT_RESOURCES;
            }

            if (m_bCaseInsensitive)
            {
                SwarUpcaseBuffer(pszCopy, pszPattern, cchLen);
            }
            else
            {
                RtlCopyMemory(pszCopy, pszPattern, cchLen * sizeof(TChar));
            }
        }

        m_pPatterns[m_ulPatternCount].pszPattern = pszCopy;
        m_pPatterns[m_ulPatternCount].ulLength   = static_cast<ULONG>(cchLen);
        m_pPatterns[m_ulPatternCount].eScope     = eScope;
        m_pPatterns[m_ulPatternCount].Context    = rContext;
        m_ulPatternCount++;

        return STATUS_SUCCESS;
    }

    // ---------------------------------------------------------------------------------
    // Compiles all staged patterns into a prefix-deduplicated Trie NFA and builds
    // all SWAR acceleration tables.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(APC_LEVEL)
    NTSTATUS Compile() noexcept
    {
        PAGED_CODE();

        FreeCompiledBuffers();

        if (m_ulPatternCount == 0)
        {
            return STATUS_SUCCESS;
        }

        // Step 1: Insert all normalized patterns into the prefix Trie
        TRIE_NODE* pTrieNodes      = NULL;
        ULONG      ulTrieNodeCount = 0;
        NTSTATUS   ntStatus        = BuildPrefixTrie(&pTrieNodes, &ulTrieNodeCount);
        if (!NT_SUCCESS(ntStatus))
        {
            return ntStatus;
        }

        // Step 2: Assign state indices using Chain-Contiguous Sibling BFS so sibling
        // branches remain contiguous while single-child chains pack into the same word.
        ntStatus = AssignStateIndicesBfs(pTrieNodes, ulTrieNodeCount);
        if (!NT_SUCCESS(ntStatus))
        {
            ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
            FreeCompiledBuffers();
            return ntStatus;
        }

        // Step 3: Allocate and populate runtime NFA instruction, cold state metadata, and SWAR filter tables
        ULONG64* pulScratchTargetBits = NULL;
        ntStatus = AllocateRuntimeTables(pTrieNodes, ulTrieNodeCount, &pulScratchTargetBits);
        if (!NT_SUCCESS(ntStatus))
        {
            ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);
            FreeCompiledBuffers();
            return ntStatus;
        }

        PopulateInstructionsAndFilters(pTrieNodes, ulTrieNodeCount, pulScratchTargetBits);
        ExFreePoolWithTag(pulScratchTargetBits, NFA_POOL_TAG);

        // Step 4: Configure SWAR Optimization 2 (Steady-State '*' 64-Bit Fast-Forward)
        ConfigureStarSteadyStateSkip(pTrieNodes);

        // Step 5: Configure SWAR Optimization 3 (Shared Root Literal Prefix Fast-Forward)
        ConfigureRootLiteralPrefix(pTrieNodes, ulTrieNodeCount);

        // Step 6: Build unbranched Op::Literal run tables for mid-Trie SWAR fast-forwarding
        BuildLiteralRunTables(pTrieNodes, ulTrieNodeCount);

        ExFreePoolWithTag(pTrieNodes, NFA_POOL_TAG);

        // Publish compiled state with Release semantics so all populated NFA tables
        // are globally visible on ARM64 before concurrent readers observe m_bCompiled == TRUE.
        WriteBooleanRelease(&m_bCompiled, TRUE);
        return STATUS_SUCCESS;
    }

    // ---------------------------------------------------------------------------------
    // Evaluates input text against all compiled patterns and returns immediately upon
    // resolving the first matching pattern context.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(DISPATCH_LEVEL)
    BOOLEAN MatchFirst(_In_reads_or_z_(cchLen) const TChar* __restrict pszText,
                       _In_                    size_t                  cchLen,
                       _Out_opt_               TContext* __restrict    pOutContext) const noexcept
    {
        if (m_bCaseInsensitive)
        {
            return ExecuteVm<TRUE>(pszText, cchLen, pOutContext, 1, NULL, TRUE);
        }
        else
        {
            return ExecuteVm<FALSE>(pszText, cchLen, pOutContext, 1, NULL, TRUE);
        }
    }

    // ---------------------------------------------------------------------------------
    // Evaluates input text against all compiled patterns and populates pOutContexts
    // with all matching pattern contexts up to ulMaxMatches.
    // ---------------------------------------------------------------------------------
    _IRQL_requires_max_(DISPATCH_LEVEL)
    BOOLEAN MatchAll(_In_reads_or_z_(cchLen)        const TChar* __restrict pszText,
                     _In_                           size_t                  cchLen,
                     _Out_writes_opt_(ulMaxMatches) TContext* __restrict    pOutContexts,
                     _In_                           ULONG                   ulMaxMatches,
                     _Out_opt_                      ULONG* __restrict       pulMatchCount) const noexcept
    {
        if (m_bCaseInsensitive)
        {
            return ExecuteVm<TRUE>(pszText, cchLen, pOutContexts, ulMaxMatches, pulMatchCount, FALSE);
        }
        else
        {
            return ExecuteVm<FALSE>(pszText, cchLen, pOutContexts, ulMaxMatches, pulMatchCount, FALSE);
        }
    }
};

// Convenience type aliases matching KmStringPatternMatchW / KmStringPatternMatchA
template <typename TContext>
using KmWildcardNfaW = KmWildcardNfa<TContext, WCHAR>;

template <typename TContext>
using KmWildcardNfaA = KmWildcardNfa<TContext, CHAR>;