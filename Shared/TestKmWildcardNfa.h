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

// -------------------------------------------------------------------------------------
// Unified Test Logic for KmWildcardNfa (Kernel & User Mode)
// -------------------------------------------------------------------------------------
// Test Suite & Benchmark Architecture Overview:
// This file implements a comprehensive kernel and user-mode unit test and benchmarking
// harness for the KmWildcardNfa engine. It validates functional correctness across
// 8-bit CHAR and 16-bit WCHAR string pipelines, wildcard scopes, escapes, and SWAR
// optimizations, while providing high-resolution multi-threaded performance benchmarks
// pinned to physical P-cores.
// -------------------------------------------------------------------------------------
#pragma once

#define TEST_POOL__TAG 'WCMp'

extern volatile LONG g_lAbortTests;

VOID FlushLogToFile();

#define MAX_TEST_THREADS 64

// Orders all prior loads and stores before any subsequent loads and stores.
__forceinline void KmReadWriteBarrier() noexcept
{
#if defined(_M_ARM64) || defined(_M_ARM64EC)
    __dmb(_ARM64_BARRIER_ISH);
#else
    _ReadWriteBarrier();
#endif
}

// Configuration structure allowing callers to selectively run modules
struct TEST_SUITE_CONFIG
{
    BOOLEAN bRunCorrectness; // Flag to execute the functional correctness test suite
    BOOLEAN bRunPerformance; // Flag to execute the high-throughput performance test suite
};

// Context tracking state for individual test worker threads
struct TEST_WORKER_CONTEXT
{
    ULONG          ulThreadId;    // Unique identifier assigned to the worker thread
    PKEVENT        pStartEvent;   // Kernel event used to synchronize thread startup
    volatile LONG* plStopFlag;    // Shared flag signaling all threads to safely terminate
    PVOID          pUserContext;  // Opaque pointer to user-defined test state data
    BOOLEAN        bSetAffinity;  // Flag to apply explicit thread affinity locking
    USHORT         AffinityGroup; // Processor group assigned for thread affinity
    KAFFINITY      AffinityMask;  // Logical processor mask assigned for thread affinity
};

typedef VOID(*PTEST_WORKER_FUNC)(_Inout_ TEST_WORKER_CONTEXT* pContext);

// Manager handling thread pooling and synchronization
struct TEST_THREAD_MANAGER
{
    PETHREAD            pThreads[MAX_TEST_THREADS]; // Array of kernel thread object pointers
    TEST_WORKER_CONTEXT Contexts[MAX_TEST_THREADS]; // Pre-allocated contexts for each worker
    ULONG               ulThreadCount;              // Total number of actively running threads
    KEVENT              StartEvent;                 // Master event to unblock all workers simultaneously
    volatile LONG       lStopFlag;                  // Master termination signal flag for the pool
};

// -------------------------------------------------------------------------------------
// Processor Topology & Robust P-Core Detector (Excludes E-cores and LP-cores)
// Used to prevent thread throttling and context switching during performance tiers.
// -------------------------------------------------------------------------------------
struct PCORE_THREAD_INFO
{
    USHORT    Group;      // Processor group index
    KAFFINITY ThreadMask; // Isolated 1-bit mask representing a distinct logical processor on a P-core
};

inline ULONG GetPCoreLogicalThreads(_Out_writes_(ulMaxCores) PCORE_THREAD_INFO* pCoreThreads, 
                                    _In_                     ULONG              ulMaxCores)
{
    ULONG ulCount = 0;
    ULONG length  = 0;

    NTSTATUS status = KeQueryLogicalProcessorRelationship(NULL, RelationProcessorCore, NULL, &length);
    if (status != STATUS_BUFFER_TOO_SMALL && length == 0)
    {
        return 0;
    }

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = 
        static_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ExAllocatePool2(POOL_FLAG_PAGED, length, TEST_POOL__TAG));

    if (!buffer)
    {
        return 0;
    }

    status = KeQueryLogicalProcessorRelationship(NULL, RelationProcessorCore, buffer, &length);
    if (!NT_SUCCESS(status))
    {
        ExFreePoolWithTag(buffer, TEST_POOL__TAG);
        return 0;
    }

    UCHAR   maxEfficiencyClass = 0;
    BOOLEAN foundCore          = FALSE;
    ULONG   offset             = 0;

    while (offset < length)
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<PUCHAR>(buffer) + offset);

        if (info->Relationship == RelationProcessorCore)
        {
            foundCore = TRUE;
            if (info->Processor.EfficiencyClass > maxEfficiencyClass)
            {
                maxEfficiencyClass = info->Processor.EfficiencyClass;
            }
        }

        if (info->Size == 0)
        {
            break;
        }

        offset += info->Size;
    }

    if (!foundCore)
    {
        ExFreePoolWithTag(buffer, TEST_POOL__TAG);
        return 0;
    }

    offset = 0;
    while (offset < length && ulCount < ulMaxCores)
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<PUCHAR>(buffer) + offset);

        if (info->Relationship == RelationProcessorCore && info->Processor.EfficiencyClass == maxEfficiencyClass)
        {
            for (USHORT g = 0; g < info->Processor.GroupCount && ulCount < ulMaxCores; ++g)
            {
                KAFFINITY coreMask = info->Processor.GroupMask[g].Mask;
                USHORT    group    = info->Processor.GroupMask[g].Group;

                // Unpack each logical SMT/HT thread bit belonging to this physical P-core
                while (coreMask != 0 && ulCount < ulMaxCores)
                {
                    KAFFINITY singleThreadMask = coreMask & (~coreMask + 1);
                    coreMask &= ~singleThreadMask;

                    pCoreThreads[ulCount].Group      = group;
                    pCoreThreads[ulCount].ThreadMask = singleThreadMask;
                    ulCount++;
                }
            }
        }

        if (info->Size == 0)
        {
            break;
        }

        offset += info->Size;
    }

    ExFreePoolWithTag(buffer, TEST_POOL__TAG);
    return ulCount;
}

// -------------------------------------------------------------------------------------
// String Length Helpers abstracting char/WCHAR native operations
// -------------------------------------------------------------------------------------
inline size_t KStringLength(_In_z_ const char* __restrict s)
{
    return strlen(s);
}

inline size_t KStringLength(_In_z_ const wchar_t* __restrict s)
{
    return wcslen(s);
}

// -------------------------------------------------------------------------------------
// Initializes and starts a pool of worker threads synchronized on an event.
// Automatically pins threads to logical P-core threads to eliminate preemption.
// -------------------------------------------------------------------------------------
VOID StartThreads(_Inout_  TEST_THREAD_MANAGER* pMgr, 
                  _In_     ULONG                ulCount, 
                  _In_     PTEST_WORKER_FUNC    pFunc, 
                  _In_opt_ PVOID                pUserContext)
{
    PAGED_CODE();

    PCORE_THREAD_INFO pCoreThreads[MAX_TEST_THREADS];
    ULONG             ulCoreCount = GetPCoreLogicalThreads(pCoreThreads, MAX_TEST_THREADS);

    if (ulCoreCount > 0)
    {
        if (ulCount > ulCoreCount)
        {
            ulCount = ulCoreCount;
        }
    }
    else
    {
        ULONG activeProcs = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);
        if (activeProcs > 0 && ulCount > activeProcs)
        {
            ulCount = activeProcs;
        }
    }

    if (ulCount > MAX_TEST_THREADS) [[unlikely]]
    {
        ulCount = MAX_TEST_THREADS;
    }

    pMgr->ulThreadCount = ulCount;
    WriteNoFence(&pMgr->lStopFlag, 0);
    
    KeInitializeEvent(&pMgr->StartEvent, NotificationEvent, FALSE);

    for (ULONG ulIndex = 0; ulIndex < ulCount; ++ulIndex)
    {
        pMgr->Contexts[ulIndex].ulThreadId   = ulIndex;
        pMgr->Contexts[ulIndex].pStartEvent  = &pMgr->StartEvent;
        pMgr->Contexts[ulIndex].plStopFlag   = &pMgr->lStopFlag;
        pMgr->Contexts[ulIndex].pUserContext = pUserContext;

        if (ulCoreCount > 0 && ulIndex < ulCoreCount)
        {
            pMgr->Contexts[ulIndex].bSetAffinity  = TRUE;
            pMgr->Contexts[ulIndex].AffinityGroup = pCoreThreads[ulIndex].Group;
            pMgr->Contexts[ulIndex].AffinityMask  = pCoreThreads[ulIndex].ThreadMask;
        }
        else
        {
            pMgr->Contexts[ulIndex].bSetAffinity  = FALSE;
            pMgr->Contexts[ulIndex].AffinityGroup = 0;
            pMgr->Contexts[ulIndex].AffinityMask  = 0;
        }

        HANDLE   hThread;
        NTSTATUS ntStatus = PsCreateSystemThread(&hThread, 
                                                 0, 
                                                 NULL, 
                                                 NULL, 
                                                 NULL, 
                                                 reinterpret_cast<PKSTART_ROUTINE>(pFunc),
                                                 &pMgr->Contexts[ulIndex]);
        if (NT_SUCCESS(ntStatus)) [[likely]]
        {
            ntStatus = ObReferenceObjectByHandle(hThread, 
                                                 0, 
                                                 NULL, 
                                                 KernelMode, 
                                                 reinterpret_cast<PVOID*>(&pMgr->pThreads[ulIndex]), 
                                                 NULL);                                                 
            if (NT_SUCCESS(ntStatus)) [[likely]]
            {
                KeSetPriorityThread(pMgr->pThreads[ulIndex], 15);
                ZwClose(hThread);
            }
            else
            {
                pMgr->pThreads[ulIndex] = NULL;
                WriteRelease(&pMgr->lStopFlag, 1);
                WriteRelease(&g_lAbortTests, 1);

                KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);
                KeWaitForSingleObject(hThread, Executive, KernelMode, FALSE, NULL);
                ZwClose(hThread);
            }
        }
        else
        {
            pMgr->pThreads[ulIndex] = NULL;
        }
    }
    
    KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);
}

// -------------------------------------------------------------------------------------
// Signals threads to stop and waits for them to terminate safely.
// -------------------------------------------------------------------------------------
VOID StopAndWaitThreads(_Inout_ TEST_THREAD_MANAGER* pMgr, 
                        _In_    int                  nSleepSeconds)
{
    PAGED_CODE();

    if (nSleepSeconds > 0)
    {
        int nIterations = nSleepSeconds * 100;

        for (int nIndex = 0; nIndex < nIterations; ++nIndex)
        {
            if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
            {
                break;
            }
            
            LARGE_INTEGER liDelay;
            liDelay.QuadPart = -100000LL; 
            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);
        }
    }

    WriteRelease(&pMgr->lStopFlag, 1);

    for (ULONG ulIndex = 0; ulIndex < pMgr->ulThreadCount; ++ulIndex)
    {
        if (pMgr->pThreads[ulIndex] != NULL)
        {
            KeWaitForSingleObject(pMgr->pThreads[ulIndex], Executive, KernelMode, FALSE, NULL);
            ObDereferenceObject(pMgr->pThreads[ulIndex]);
        }
    }
}

// -------------------------------------------------------------------------------------
// NFA Target Context
// -------------------------------------------------------------------------------------
struct PatternContext
{
    ULONG ulPatternId;
};

// -------------------------------------------------------------------------------------
// Test Function: RunSingleCorrectnessTest
// Validates functional correctness of a single wildcard pattern and text pair across
// both MatchAll and MatchFirst evaluation pipelines.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunSingleCorrectnessTest(_In_ const TChar* __restrict pszText,
                                 _In_ const TChar* __restrict pszPattern,
                                 _In_ UINT8                   ucScope,
                                 _In_ BOOLEAN                 bShouldMatch,
                                 _In_ BOOLEAN                 bCaseInsensitive,
                                 _In_ const char*             pszTestName)
{
    PAGED_CODE();

    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        return FALSE;
    }

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    PatternContext                       ctx = { 100 };
    
    if (!NT_SUCCESS(Nfa.AddPattern(pszPattern, KStringLength(pszPattern), static_cast<typename KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE>(ucScope), ctx))) [[unlikely]]
    {
        LOG_ERR("[NFA_TEST] [!] %s - AddPattern Failed\n", pszTestName);
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        LOG_ERR("[NFA_TEST] [!] %s - Compile Failed\n", pszTestName);
        return FALSE;
    }

    PatternContext matchedCtxAll[2];
    ULONG          ulMatchCount = 0;
    BOOLEAN        bMatchedAll  = Nfa.MatchAll(pszText, KStringLength(pszText), matchedCtxAll, 2, &ulMatchCount);

    PatternContext matchedCtxFirst;
    BOOLEAN        bMatchedFirst = Nfa.MatchFirst(pszText, KStringLength(pszText), &matchedCtxFirst);

    if (bMatchedAll == bShouldMatch && bMatchedFirst == bShouldMatch) [[likely]]
    {
        LOG_INFO("[NFA_TEST] [+] PASS: %s\n", pszTestName);
        return TRUE;
    }
    
    LOG_ERR("[NFA_TEST] [!] FAIL: %s (Expected %d, Got MatchAll: %d, MatchFirst: %d)\n", pszTestName, bShouldMatch, bMatchedAll, bMatchedFirst);
    return FALSE;
}

// -------------------------------------------------------------------------------------
// Test Function: RunMultiPatternCorrectnessTest
// Validates overlapping rules to ensure MatchFirst properly extracts a single context
// among multiple registered patterns.
// -------------------------------------------------------------------------------------
template <typename TChar> BOOLEAN RunMultiPatternCorrectnessTest(_In_ const TChar* __restrict pszPattern1,
                                                                 _In_ const TChar* __restrict pszPattern2,
                                                                 _In_ const TChar* __restrict pszText,
                                                                 _In_ BOOLEAN                 bCaseInsensitive)
{
    PAGED_CODE();

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    PatternContext                       ctx1 = { 1 };
    PatternContext                       ctx2 = { 2 };
    
    if (!NT_SUCCESS(Nfa.AddPattern(pszPattern1, KStringLength(pszPattern1), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx1))) [[unlikely]]
    {
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.AddPattern(pszPattern2, KStringLength(pszPattern2), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx2))) [[unlikely]]
    {
        return FALSE;
    }
    
    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        return FALSE;
    }
    
    PatternContext results[2];
    ULONG          ulCount = 0;
    BOOLEAN        bAll    = Nfa.MatchAll(pszText, KStringLength(pszText), results, 2, &ulCount);
    
    PatternContext firstRes = { 0 };
    BOOLEAN        bFirst   = Nfa.MatchFirst(pszText, KStringLength(pszText), &firstRes);
    
    if (bAll && ulCount == 2 && bFirst && (firstRes.ulPatternId == 1 || firstRes.ulPatternId == 2)) [[likely]]
    {
        LOG_INFO("[NFA_TEST] [+] PASS: Multi-Pattern MatchAll & MatchFirst Overlap\n");
        return TRUE;
    }
    
    LOG_ERR("[NFA_TEST] [!] FAIL: Multi-Pattern MatchAll & MatchFirst Overlap\n");
    return FALSE;
}

// -------------------------------------------------------------------------------------
// Test Function: RunMatchAllSeveralPatternsTest
// Validates that MatchAll successfully matches several overlapping patterns concurrently
// and resolves all expected pattern contexts.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunMatchAllSeveralPatternsTest(_In_ const TChar* __restrict pat1,
                                       _In_ const TChar* __restrict pat2,
                                       _In_ const TChar* __restrict pat3,
                                       _In_ const TChar* __restrict pat4,
                                       _In_ const TChar* __restrict text,
                                       _In_ BOOLEAN                 bCaseInsensitive)
{
    PAGED_CODE();

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    PatternContext                       ctx1 = { 10 };
    PatternContext                       ctx2 = { 20 };
    PatternContext                       ctx3 = { 30 };
    PatternContext                       ctx4 = { 40 };

    if (!NT_SUCCESS(Nfa.AddPattern(pat1, KStringLength(pat1), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx1))) [[unlikely]]
    {
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.AddPattern(pat2, KStringLength(pat2), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx2))) [[unlikely]]
    {
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.AddPattern(pat3, KStringLength(pat3), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx3))) [[unlikely]]
    {
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.AddPattern(pat4, KStringLength(pat4), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx4))) [[unlikely]]
    {
        return FALSE;
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        return FALSE;
    }

    PatternContext results[5];
    ULONG          ulCount = 0;
    BOOLEAN        bAll    = Nfa.MatchAll(text, KStringLength(text), results, 5, &ulCount);

    if (bAll && (ulCount == 3)) [[likely]]
    {
        BOOLEAN bHas1 = FALSE;
        BOOLEAN bHas2 = FALSE;
        BOOLEAN bHas4 = FALSE;

        for (ULONG i = 0; i < ulCount; i++)
        {
            if (results[i].ulPatternId == 10)
            {
                bHas1 = TRUE;
            }

            if (results[i].ulPatternId == 20)
            {
                bHas2 = TRUE;
            }

            if (results[i].ulPatternId == 40)
            {
                bHas4 = TRUE;
            }
        }

        if (bHas1 && bHas2 && bHas4)
        {
            LOG_INFO("[NFA_TEST] [+] PASS: MatchAll Successfully Resolved Several Overlapping Patterns\n");
            return TRUE;
        }
    }

    LOG_ERR("[NFA_TEST] [!] FAIL: MatchAll Failed to Resolve Multiple Expected Patterns\n");
    return FALSE;
}

// -------------------------------------------------------------------------------------
// Test Function: RunEmptyEngineTest
// Validates the engine correctly handles queries when zero patterns are registered,
// executing the early O(1) fast-fail branch.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunEmptyEngineTest(_In_ const TChar* __restrict pszText,
                           _In_ BOOLEAN                 bCaseInsensitive)
{
    PAGED_CODE();

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);

    PatternContext matchedCtxFirst;
    BOOLEAN        bMatchedFirst = Nfa.MatchFirst(pszText, KStringLength(pszText), &matchedCtxFirst);

    if (!bMatchedFirst) [[likely]]
    {
        LOG_INFO("[NFA_TEST] [+] PASS: Empty Engine Fast-Fail\n");
        return TRUE;
    }

    LOG_ERR("[NFA_TEST] [!] FAIL: Empty Engine Fast-Fail (Unexpected Match)\n");
    return FALSE;
}

// -------------------------------------------------------------------------------------
// Test Function: RunClearLifecycleTest
// Validates explicit engine state reset via Clear(), ensuring memory arenas 
// and vectors safely flush state outside of normal destruction.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunClearLifecycleTest(_In_ const TChar* __restrict pszPattern,
                              _In_ const TChar* __restrict pszText,
                              _In_ BOOLEAN                 bCaseInsensitive)
{
    PAGED_CODE();

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    PatternContext                       ctx = { 100 };

    if (!NT_SUCCESS(Nfa.AddPattern(pszPattern, KStringLength(pszPattern), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx))) [[unlikely]]
    {
        return FALSE;
    }
    
    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        return FALSE;
    }

    PatternContext matchedCtx;
    if (!Nfa.MatchFirst(pszText, KStringLength(pszText), &matchedCtx)) [[unlikely]]
    {
        return FALSE;
    }

    Nfa.Clear();

    if (!Nfa.MatchFirst(pszText, KStringLength(pszText), &matchedCtx)) [[likely]]
    {
        LOG_INFO("[NFA_TEST] [+] PASS: Engine Clear Lifecycle\n");
        return TRUE;
    }

    LOG_ERR("[NFA_TEST] [!] FAIL: Engine Clear Lifecycle (Ghost Match)\n");
    return FALSE;
}

// -------------------------------------------------------------------------------------
// Macros dynamically wrapping string literals based on the character type (CHAR vs WCHAR)
// -------------------------------------------------------------------------------------
#define WIDEN_STRING(x) L##x
#define NARROW_STRING(x) x

// -------------------------------------------------------------------------------------
// Master macro preventing code duplication for exhaustive correctness validation
// -------------------------------------------------------------------------------------
#define EXECUTE_CORRECTNESS_SUITE(CharType, STR) \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("file.txt"), STR("file.txt"), 0, TRUE, TRUE, "Exact Literal Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("file.txt"), STR("*.txt"), 0, TRUE, TRUE, "Wildcard Prefix Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("prefix_file.txt"), STR("prefix*"), 0, TRUE, TRUE, "Wildcard Suffix Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("my_cool_file.txt"), STR("*cool*"), 0, TRUE, TRUE, "Wildcard Middle Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a"), STR("?"), 0, TRUE, TRUE, "Single Any (?) Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("ab"), STR("?"), 0, FALSE, TRUE, "Single Any (?) Mismatch on Length")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abc"), STR("a?c"), 0, TRUE, TRUE, "Single Any (?) Middle")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("Upper.TXT"), STR("upper.txt"), 0, TRUE, TRUE, "Case Insensitivity Check (Match)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("C:\\Temp\\Sub\\file.txt"), STR("C:\\*\\file.txt"), 0, TRUE, TRUE, "Wildcard Default Scope (Allows Slash)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("C:\\Temp\\Sub\\file.txt"), STR("C:\\*\\file.txt"), 1, FALSE, TRUE, "Wildcard Slash Scope (Blocks Slash)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("Upper.TXT"), STR("upper.txt"), 0, FALSE, FALSE, "Case Sensitivity Check (Mismatch)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("ExactCase.TXT"), STR("ExactCase.TXT"), 0, TRUE, FALSE, "Case Sensitivity Check (Exact Match)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("ExactStringTestMatch"), STR("ExactStringTestMatch"), 0, TRUE, TRUE, "Exact String Check (Match)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("ExactStringTestMatch"), STR("ExactStringMismatch"), 0, FALSE, TRUE, "Exact String Check (Mismatch)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a_z_test"), STR("A_Z_TEST"), 0, TRUE, TRUE, "SWAR Boundary Transformation ('a' and 'z')")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("`a{z`a{z"), STR("`A{Z`A{Z"), 0, TRUE, TRUE, "SWAR Adjacent Exclusion ('`' and '{')")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abcdefgh"), STR("ABCDEFGH"), 0, TRUE, TRUE, "SWAR Full Block Transformation")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("!@#$[]{}"), STR("!@#$[]{}"), 0, TRUE, TRUE, "SWAR Full Block Exclusion")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR(""), STR(""), 0, TRUE, TRUE, "Zero-Length Exact Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR(""), STR("*"), 0, TRUE, TRUE, "Zero-Length Wildcard Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a"), STR(""), 0, FALSE, TRUE, "Zero-Length Pattern Mismatch")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a"), STR("a*"), 0, TRUE, TRUE, "Suffix Wildcard on Single Char")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abcdef"), STR("a*f"), 0, TRUE, TRUE, "Boundary Wildcard Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abcdef"), STR("a*z"), 0, FALSE, TRUE, "Boundary Wildcard Mismatch")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abc"), STR("????"), 0, FALSE, TRUE, "Too Many Wildcards (?)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abc"), STR("??"), 0, FALSE, TRUE, "Too Few Wildcards (?)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a1b2c3d4"), STR("a?b?c?d?"), 0, TRUE, TRUE, "Alternating Wildcards (?)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("hello"), STR("**hello**"), 0, TRUE, TRUE, "Redundant Asterisks (*)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("hello"), STR("*h*e*l*l*o*"), 0, TRUE, TRUE, "Scattered Asterisks (*)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("C:\\Temp\\Sub\\test.txt"), STR("C:\\*\\*.txt"), 1, FALSE, TRUE, "Strict Path Wildcard Multiple Slashes (Slash Scope)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("C:\\Temp\\Sub\\test.txt"), STR("C:\\*\\*.txt"), 0, TRUE, TRUE, "Path Wildcard Multiple Slashes (Default Scope)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a*b"), STR("a|*b"), 0, TRUE, TRUE, "Escape Character (Escaped *)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a?b"), STR("a|?b"), 0, TRUE, TRUE, "Escape Character (Escaped ?)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a|b"), STR("a||b"), 0, TRUE, TRUE, "Escape Character (Escaped |)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a*b"), STR("a|?b"), 0, FALSE, TRUE, "Escape Character (Mismatch on Escaped)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abc"), STR("a|*c"), 0, FALSE, TRUE, "Escape Character (Literal mismatch against wildcard)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("abc|"), STR("abc|"), 0, TRUE, TRUE, "Escape Character (Dangling Escape)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a|x"), STR("a|x"), 0, TRUE, TRUE, "Escape Character (Unrecognized Escape acts as literal)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a|*b"), STR("a|||*b"), 0, TRUE, TRUE, "Escape Character (Consecutive Interleaved Escapes)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("A*B"), STR("a|*b"), 0, TRUE, TRUE, "Escape Character (Case-Insensitive Escaped Wildcard)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("axb"), STR("a*?b"), 0, TRUE, TRUE, "Mixed Wildcards (*?) Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("ab"), STR("a*?b"), 0, FALSE, TRUE, "Mixed Wildcards (*?) Length Mismatch")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("axyzb"), STR("a?*b"), 0, TRUE, TRUE, "Mixed Wildcards (?*) Match")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR(""), STR("?"), 0, FALSE, TRUE, "Single Any (?) against Empty String")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("a\\b"), STR("a?b"), 1, FALSE, TRUE, "Scope Slash blocks Single Any (?)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("short"), STR("muchlongerpattern"), 0, FALSE, TRUE, "Length Bounds Fast-Fail")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("123456789"), STR("123456789"), 0, TRUE, FALSE, "SWAR Unaligned Tail Check (Match)")); \
    CheckTest(RunSingleCorrectnessTest<CharType>(STR("123456789"), STR("123456780"), 0, FALSE, FALSE, "SWAR Unaligned Tail Check (Mismatch)")); \
    CheckTest(RunEmptyEngineTest<CharType>(STR("FastExitCheck"), TRUE)); \
    CheckTest(RunClearLifecycleTest<CharType>(STR("flush.txt"), STR("flush.txt"), TRUE)); \
    CheckTest(RunMultiPatternCorrectnessTest<CharType>(STR("*.txt"), STR("file.txt"), STR("file.txt"), TRUE)); \
    CheckTest(RunMatchAllSeveralPatternsTest<CharType>(STR("*.txt"), STR("test.*"), STR("*.log"), STR("te*xt"), STR("test.txt"), TRUE));

// -------------------------------------------------------------------------------------
// Validates truth table against multiple wildcard constraints for both types.
// -------------------------------------------------------------------------------------
VOID RunAllCorrectnessTests()
{
    PAGED_CODE();

    ULONG ulPassed = 0;
    ULONG ulFailed = 0;

    auto CheckTest = [&](BOOLEAN bResult)
    {
        if (bResult) [[likely]]
        {
            ulPassed++;
        }
        else
        {
            ulFailed++;
        }
    };

    LOG_INFO("\n[NFA_TEST] --- Correctness Tests [WCHAR] ---\n");
    EXECUTE_CORRECTNESS_SUITE(WCHAR, WIDEN_STRING);

    LOG_INFO("\n[NFA_TEST] --- Correctness Tests [CHAR] ---\n");
    EXECUTE_CORRECTNESS_SUITE(CHAR, NARROW_STRING);
    
    if (ulFailed == 0) [[likely]]
    {
        LOG_INFO("\n[NFA_TEST] [+] ALL TESTS SUCCEEDED (%u passed)\n", ulPassed);
    }
    else
    {
        LOG_ERR("\n[NFA_TEST] [!] SOME TESTS FAILED (%u passed, %u failed)\n", ulPassed, ulFailed);
    }
}

// -------------------------------------------------------------------------------------
// Data packet fed into templated threaded performance workers.
// -------------------------------------------------------------------------------------
template <typename TChar>
struct PerfWorkerCtx
{
    KmWildcardNfa<PatternContext, TChar>* pNfa;             // Pointer to the active NFA instance
    const TChar*                          pText;            // Pointer to the target text buffer to search
    size_t                                cchTextLen;       // Length of the text buffer in characters
    volatile LONG64                       llTotalOpsPerSec; // Pre-aggregated rate of operations across all threads
    LARGE_INTEGER                         liFreq;           // High-resolution clock frequency provided by orchestrator
    volatile LONG                         lStartFlag;       // Flag indicating the worker has formally started
    BOOLEAN                               bUseMatchFirst;   // Toggles between MatchFirst and MatchAll modes
    BOOLEAN                               bExpectMatch;     // Indicates whether a match is logically expected
    volatile LONG                         lTestFailed;      // Flag set if a correctness logic violation occurs
    volatile LONG                         lActualResult;    // Captures the erroneous boolean result on failure
};

// -------------------------------------------------------------------------------------
// Iteratively runs search operations in a tight loop to calculate throughput.
// Branches loop internally to prevent branch prediction overhead during measurement.
// Terminates loop and safely notifies orchestrator if an unexpected match logic failure occurs.
// Applies thread affinity directly to the kernel thread scope if executing natively.
// -------------------------------------------------------------------------------------
template <typename TChar>
VOID PerfWorkerT(_Inout_ TEST_WORKER_CONTEXT* __restrict pCtx)
{
    PAGED_CODE();

    if (pCtx->bSetAffinity)
    {
        GROUP_AFFINITY affinity = { 0 };
        affinity.Group = pCtx->AffinityGroup;
        affinity.Mask  = pCtx->AffinityMask;
        KeSetSystemGroupAffinityThread(&affinity, NULL);
    }

    PerfWorkerCtx<TChar>* pWorkerCtx = static_cast<PerfWorkerCtx<TChar>*>(pCtx->pUserContext);
    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    UINT64  localOps = 0;
    BOOLEAN bAborted = FALSE;

    // Spin with Acquire semantics so ARM64 observes all prior context writes before starting
    while (ReadAcquire(&pWorkerCtx->lStartFlag) == 0)
    {
        if (ReadNoFence(&g_lAbortTests) != 0) [[unlikely]]
        {
            bAborted = TRUE;
            break;
        }
        
        YieldProcessor(); // Mitigates thread start skew during metrics capture
    }

    if (!bAborted) [[likely]]
    {
        // 1. Capture exact start time immediately before the work loop
        LARGE_INTEGER liStart = KeQueryPerformanceCounter(NULL);

        if (pWorkerCtx->bUseMatchFirst)
        {
            PatternContext Result;
            
            // Use ReadNoFence inside the hot loop to avoid atomic RMW pipeline stalls on ARM64/x64
            while (ReadNoFence(pCtx->plStopFlag) == 0 &&
                   ReadNoFence(&g_lAbortTests) == 0) [[likely]]
            {
                BOOLEAN bRet = pWorkerCtx->pNfa->MatchFirst(pWorkerCtx->pText, pWorkerCtx->cchTextLen, &Result);
                
                // Assert throughput correctness - instantly abort thread and flag for shutdown on logic failure
                if (bRet != pWorkerCtx->bExpectMatch) [[unlikely]]
                {
                    WriteNoFence(&pWorkerCtx->lActualResult, bRet ? 1 : 0);
                    WriteRelease(&pWorkerCtx->lTestFailed, 1);
                    break;
                }
                
                localOps++;
                KmReadWriteBarrier();
            }
        }
        else
        {
            PatternContext Results[10];
            ULONG          ulMatched;
            
            // Use ReadNoFence inside the hot loop to avoid atomic RMW pipeline stalls on ARM64/x64
            while (ReadNoFence(pCtx->plStopFlag) == 0 &&
                   ReadNoFence(&g_lAbortTests) == 0) [[likely]]
            {
                BOOLEAN bRet = pWorkerCtx->pNfa->MatchAll(pWorkerCtx->pText, pWorkerCtx->cchTextLen, Results, 10, &ulMatched);
                
                // Assert throughput correctness - instantly abort thread and flag for shutdown on logic failure
                if (bRet != pWorkerCtx->bExpectMatch) [[unlikely]]
                {
                    WriteNoFence(&pWorkerCtx->lActualResult, bRet ? 1 : 0);
                    WriteRelease(&pWorkerCtx->lTestFailed, 1);
                    break;
                }
                
                localOps++;
                KmReadWriteBarrier();
            }
        }

        // 2. Capture exact end time immediately after the stop flag is detected
        LARGE_INTEGER liEnd = KeQueryPerformanceCounter(NULL);

        // 3. Calculate this thread's exact ops/sec, completely excluding OS teardown overhead
        UINT64 localTicks = liEnd.QuadPart - liStart.QuadPart;
        
        if (localTicks > 0 && localOps > 0)
        {
            UINT64 localOpsPerSec = (localOps * pWorkerCtx->liFreq.QuadPart) / localTicks;
            InterlockedExchangeAdd64(&pWorkerCtx->llTotalOpsPerSec, static_cast<LONG64>(localOpsPerSec));
        }
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

// -------------------------------------------------------------------------------------
// Test Function: RunPerformanceTier
// Orchestrates performance execution, spins up worker threads, and calculates throughput
// and latency metrics against pathological wildcard pattern workloads.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunPerformanceTier(_In_ ULONG       ulPatternCount,
                           _In_ ULONG       ulTextLenChars,
                           _In_ BOOLEAN     bPathological,
                           _In_ BOOLEAN     bExpectMatch,
                           _In_ BOOLEAN     bCaseInsensitive,
                           _In_ const char* pszTestName,
                           _In_ const char* pszDescription,
                           _In_ const char* pszPatternType,
                           _In_ const char* pszCharType)
{
    PAGED_CODE();
    
    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        return FALSE;
    }

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    BOOLEAN                              bTierSuccess = TRUE;
    
    const SIZE_T cbTextMap = static_cast<SIZE_T>(ulTextLenChars) * sizeof(TChar) + sizeof(TChar);
    TChar*       pwszText  = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, cbTextMap, TEST_POOL__TAG));
    
    if (!pwszText) [[unlikely]]
    {
        return FALSE;
    }

    for (ULONG i = 0; i < ulTextLenChars; i++)
    {
        if (bExpectMatch)
        {
            pwszText[i] = static_cast<TChar>('A' + (i % 26));
        }
        else
        {
            pwszText[i] = static_cast<TChar>('X'); // Forces negative matches
        }
    }
    
    // Inject a guaranteed match for Pattern 0 near the end of the text
    if (bExpectMatch && ulTextLenChars >= 16)
    {
        if (bPathological)
        {
            pwszText[ulTextLenChars - 4] = static_cast<TChar>('A');
            pwszText[ulTextLenChars - 3] = static_cast<TChar>('0');
            pwszText[ulTextLenChars - 2] = static_cast<TChar>('B');
        }
        else
        {
            pwszText[ulTextLenChars - 9] = static_cast<TChar>('S');
            pwszText[ulTextLenChars - 8] = static_cast<TChar>('Y');
            pwszText[ulTextLenChars - 7] = static_cast<TChar>('S');
            pwszText[ulTextLenChars - 6] = static_cast<TChar>('0');
            pwszText[ulTextLenChars - 5] = static_cast<TChar>('0');
            pwszText[ulTextLenChars - 4] = static_cast<TChar>('0');
            pwszText[ulTextLenChars - 3] = static_cast<TChar>('0');
            pwszText[ulTextLenChars - 2] = static_cast<TChar>('B');
        }
    }
    
    pwszText[ulTextLenChars] = static_cast<TChar>('\0');

    TChar* pStrArray = 
        static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, static_cast<SIZE_T>(ulPatternCount) * 64 * sizeof(TChar), TEST_POOL__TAG));

    if (!pStrArray) [[unlikely]]
    {
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    for (ULONG i = 0; i < ulPatternCount; i++)
    {
        if (bPathological)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(&pStrArray[i * 64]), 64, "*A%dB*", static_cast<int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(&pStrArray[i * 64]), 64, L"*A%dB*", static_cast<int>(i));
            }
        }
        else
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(&pStrArray[i * 64]), 64, "SYS%04d*B*", static_cast<int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(&pStrArray[i * 64]), 64, L"SYS%04d*B*", static_cast<int>(i));
            }
        }
        
        PatternContext ctx = { i };
        
        if (!NT_SUCCESS(Nfa.AddPattern(&pStrArray[i * 64], KStringLength(&pStrArray[i * 64]), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx))) [[unlikely]]
        {
            ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
            ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
            return FALSE;
        }
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    LOG_INFO("\n[NFA_TEST] --- Perf Test [%s]: %s [%u Patterns | Text: %u Chars] ---\n", pszCharType, pszTestName, ulPatternCount, ulTextLenChars);
    LOG_INFO("[NFA_TEST]      %s\n", pszDescription);
    LOG_INFO("[NFA_TEST]      Pattern Type: %s\n", pszPatternType);
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");
    LOG_INFO("[NFA_TEST]      %-18s | %-15s | %-15s | %-15s\n", "Engine Mode", "Throughput", "Ops/sec", "Latency/Op");
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");

    LARGE_INTEGER liFreq;
    KeQueryPerformanceCounter(&liFreq);

    auto RunMode = [&](BOOLEAN bMatchFirst, const char* label) -> BOOLEAN
    {
        if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
        {
            return FALSE;
        }

        PerfWorkerCtx<TChar> Ctx = { 0 };
        Ctx.pNfa                 = &Nfa;
        Ctx.pText                = pwszText;
        Ctx.cchTextLen           = ulTextLenChars;
        Ctx.liFreq               = liFreq;
        Ctx.llTotalOpsPerSec     = 0;
        Ctx.bUseMatchFirst       = bMatchFirst;
        Ctx.bExpectMatch         = bExpectMatch;
        Ctx.lTestFailed          = 0;

        TEST_THREAD_MANAGER* pMgr = 
            static_cast<TEST_THREAD_MANAGER*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, sizeof(TEST_THREAD_MANAGER), TEST_POOL__TAG));
        
        if (pMgr) [[likely]]
        {
            StartThreads(pMgr, 1, PerfWorkerT<TChar>, &Ctx);
                         
            LARGE_INTEGER liDelay;
            liDelay.QuadPart = -2000000LL; 
            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);

            WriteRelease(&Ctx.lStartFlag, 1);
            
            // Wait for threads to clean up
            StopAndWaitThreads(pMgr, 2);
            
            if (ReadAcquire(&Ctx.lTestFailed) != 0) [[unlikely]]
            {
                LOG_ERR("[NFA_TEST]      %-18s | [!] FAILED: Unexpected Result (Expected %d, Actual %ld)\n", label, bExpectMatch, ReadNoFence(&Ctx.lActualResult));
                
                WriteRelease(&g_lAbortTests, 1);
                
                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return FALSE;
            }
            else
            {
                UINT64 opsPerSec = static_cast<UINT64>(ReadAcquire64(&Ctx.llTotalOpsPerSec));
                
                UINT64 totalPs = opsPerSec > 0 ? ((1000000000000ULL * (pMgr->ulThreadCount > 0 ? pMgr->ulThreadCount : 1)) / opsPerSec) : 0;
                UINT64 usWhole = totalPs / 1000000;
                UINT64 usFrac  = (totalPs % 1000000) / 100;

                // Test design ensures the entire string is scanned before the match succeeds or fails.
                UINT64 bytesPerOp  = static_cast<UINT64>(Ctx.cchTextLen) * sizeof(TChar);
                UINT64 bytesPerSec = opsPerSec * bytesPerOp;
                UINT64 mbWhole     = bytesPerSec / (1024ULL * 1024ULL);

                if (mbWhole < 1)
                {
                    UINT64 mbFrac = ((bytesPerSec % (1024ULL * 1024ULL)) * 100ULL) / (1024ULL * 1024ULL);
                    LOG_INFO("[NFA_TEST]      %-18s | %7llu.%02llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, mbFrac, opsPerSec, usWhole, usFrac);
                }
                else
                {
                    LOG_INFO("[NFA_TEST]      %-18s | %10llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, opsPerSec, usWhole, usFrac);
                }

                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return TRUE;
            }
        }
        
        return FALSE;
    };

    if (RunMode(FALSE, "MatchAll")) [[likely]]
    {
        RunMode(TRUE, "MatchFirst");
    }
    else
    {
        bTierSuccess = FALSE;
    }

    ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
    ExFreePoolWithTag(pwszText, TEST_POOL__TAG);

    return bTierSuccess;
}

// -------------------------------------------------------------------------------------
// Test Function: RunRealisticPathPerformanceTier
// Benchmarks realistic kernel file path matching scenarios across large pattern sets
// and text buffers.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunRealisticPathPerformanceTier(_In_ ULONG       ulPatternCount,
                                        _In_ BOOLEAN     bExpectMatch,
                                        _In_ BOOLEAN     bCaseInsensitive,
                                        _In_ const char* pszTestName,
                                        _In_ const char* pszDescription,
                                        _In_ const char* pszPatternType,
                                        _In_ const char* pszCharType)
{
    PAGED_CODE();

    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        return FALSE;
    }

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    BOOLEAN                              bTierSuccess = TRUE;

    TChar* pStrArray = 
        static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, static_cast<SIZE_T>(ulPatternCount) * 128 * sizeof(TChar), TEST_POOL__TAG));
    
    if (!pStrArray) [[unlikely]]
    {
        return FALSE;
    }

    for (ULONG i = 0; i < ulPatternCount; i++)
    {
        TChar* pCurPat = &pStrArray[i * 128];
        if (i % 4 == 0)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(pCurPat), 128, "\\Device\\HarddiskVolume2\\Program Files\\App%u\\*\\crypt.exe", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(pCurPat), 128, L"\\Device\\HarddiskVolume2\\Program Files\\App%u\\*\\crypt.exe", static_cast<unsigned int>(i));
            }
        }
        else if (i % 4 == 1)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(pCurPat), 128, "\\Device\\HarddiskVolume2\\Windows\\System32\\module_%u.dll", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(pCurPat), 128, L"\\Device\\HarddiskVolume2\\Windows\\System32\\module_%u.dll", static_cast<unsigned int>(i));
            }
        }
        else if (i % 4 == 2)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(pCurPat), 128, "\\Device\\HarddiskVolume2\\Users\\*\\AppData\\Roaming\\payload_%u.exe", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(pCurPat), 128, L"\\Device\\HarddiskVolume2\\Users\\*\\AppData\\Roaming\\payload_%u.exe", static_cast<unsigned int>(i));
            }
        }
        else
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(pCurPat), 128, "\\Device\\HarddiskVolume2\\Temp\\malware_%u.tmp", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(pCurPat), 128, L"\\Device\\HarddiskVolume2\\Temp\\malware_%u.tmp", static_cast<unsigned int>(i));
            }
        }

        PatternContext ctx = { i };
        
        if (!NT_SUCCESS(Nfa.AddPattern(pCurPat, KStringLength(pCurPat), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx))) [[unlikely]]
        {
            ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
            return FALSE;
        }
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
        return FALSE;
    }

    // Change App42 to App0 to ensure (0 % 4 == 0) guarantees generation of a matching pattern
    const TChar* pwszTargetPath;

    if constexpr (sizeof(TChar) == 1)
    {
        pwszTargetPath = bExpectMatch 
            ? reinterpret_cast<const TChar*>("\\Device\\HarddiskVolume2\\Program Files\\App0\\bin\\crypt.exe")
            : reinterpret_cast<const TChar*>("\\Device\\HarddiskVolume2\\Program Files\\SafeApp\\bin\\clean.exe");
    }
    else
    {
        pwszTargetPath = bExpectMatch 
            ? reinterpret_cast<const TChar*>(L"\\Device\\HarddiskVolume2\\Program Files\\App0\\bin\\crypt.exe")
            : reinterpret_cast<const TChar*>(L"\\Device\\HarddiskVolume2\\Program Files\\SafeApp\\bin\\clean.exe");
    }

    size_t cchPathLen = KStringLength(pwszTargetPath);

    LOG_INFO("\n[NFA_TEST] --- Perf Test [%s]: %s [%u Patterns | Text: %u Chars] ---\n", pszCharType, pszTestName, ulPatternCount, static_cast<ULONG>(cchPathLen));
    LOG_INFO("[NFA_TEST]      %s\n", pszDescription);
    LOG_INFO("[NFA_TEST]      Pattern Type: %s\n", pszPatternType);
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");
    LOG_INFO("[NFA_TEST]      %-18s | %-15s | %-15s | %-15s\n", "Engine Mode", "Throughput", "Ops/sec", "Latency/Op");
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");

    LARGE_INTEGER liFreq;
    KeQueryPerformanceCounter(&liFreq);

    auto RunMode = [&](BOOLEAN bMatchFirst, const char* label) -> BOOLEAN
    {
        if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
        {
            return FALSE;
        }

        PerfWorkerCtx<TChar> Ctx = { 0 };
        Ctx.pNfa                 = &Nfa;
        Ctx.pText                = pwszTargetPath;
        Ctx.cchTextLen           = cchPathLen;
        Ctx.liFreq               = liFreq;
        Ctx.llTotalOpsPerSec     = 0;
        Ctx.bUseMatchFirst       = bMatchFirst;
        Ctx.bExpectMatch         = bExpectMatch;
        Ctx.lTestFailed          = 0;

        TEST_THREAD_MANAGER* pMgr = 
            static_cast<TEST_THREAD_MANAGER*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, sizeof(TEST_THREAD_MANAGER), TEST_POOL__TAG));
        
        if (pMgr) [[likely]]
        {
            StartThreads(pMgr, 1, PerfWorkerT<TChar>, &Ctx);
                         
            LARGE_INTEGER liDelay;
            liDelay.QuadPart = -2000000LL; 
            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);

            WriteRelease(&Ctx.lStartFlag, 1);
            
            // Wait for threads to clean up
            StopAndWaitThreads(pMgr, 2);
            
            if (ReadAcquire(&Ctx.lTestFailed) != 0) [[unlikely]]
            {
                LOG_ERR("[NFA_TEST]      %-18s | [!] FAILED: Unexpected Result (Expected %d, Actual %ld)\n", label, bExpectMatch, ReadNoFence(&Ctx.lActualResult));
                WriteRelease(&g_lAbortTests, 1);
                
                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return FALSE;
            }
            else
            {
                UINT64 opsPerSec = static_cast<UINT64>(ReadAcquire64(&Ctx.llTotalOpsPerSec));
                
                UINT64 totalPs = opsPerSec > 0 ? ((1000000000000ULL * (pMgr->ulThreadCount > 0 ? pMgr->ulThreadCount : 1)) / opsPerSec) : 0;
                UINT64 usWhole = totalPs / 1000000;
                UINT64 usFrac  = (totalPs % 1000000) / 100;

                // Test design ensures the entire string is scanned before the match succeeds or fails.
                UINT64 bytesPerOp  = static_cast<UINT64>(Ctx.cchTextLen) * sizeof(TChar);
                UINT64 bytesPerSec = opsPerSec * bytesPerOp;
                UINT64 mbWhole     = bytesPerSec / (1024ULL * 1024ULL);

                if (mbWhole < 1)
                {
                    UINT64 mbFrac = ((bytesPerSec % (1024ULL * 1024ULL)) * 100ULL) / (1024ULL * 1024ULL);
                    LOG_INFO("[NFA_TEST]      %-18s | %7llu.%02llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, mbFrac, opsPerSec, usWhole, usFrac);
                }
                else
                {
                    LOG_INFO("[NFA_TEST]      %-18s | %10llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, opsPerSec, usWhole, usFrac);
                }

                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return TRUE;
            }
        }
        
        return FALSE;
    };

    if (RunMode(FALSE, "MatchAll")) [[likely]]
    {
        RunMode(TRUE, "MatchFirst");
    }
    else
    {
        bTierSuccess = FALSE;
    }

    ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);

    return bTierSuccess;
}

// -------------------------------------------------------------------------------------
// Test Function: RunMeaningfulTextPerformanceTier
// Benchmarks meaningful prose context performance with multi-wildcard boundaries.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunMeaningfulTextPerformanceTier(_In_ ULONG       ulPatternCount,
                                         _In_ ULONG       ulTextLenChars,
                                         _In_ BOOLEAN     bExpectMatch,
                                         _In_ BOOLEAN     bCaseInsensitive,
                                         _In_ const char* pszTestName,
                                         _In_ const char* pszDescription,
                                         _In_ const char* pszPatternType,
                                         _In_ const char* pszCharType)
{
    PAGED_CODE();

    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        return FALSE;
    }

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    BOOLEAN                              bTierSuccess = TRUE;
    
    // Meaningful prose context to exercise realistic string density dynamically selected by char size
    const TChar* pwszMeaningfulText;
    if constexpr (sizeof(TChar) == 1)
    {
        pwszMeaningfulText = reinterpret_cast<const TChar*>("The Windows Kernel memory manager employs aggressive demand paging and lookaside lists to satisfy pool allocations efficiently. ");
    }
    else
    {
        pwszMeaningfulText = reinterpret_cast<const TChar*>(L"The Windows Kernel memory manager employs aggressive demand paging and lookaside lists to satisfy pool allocations efficiently. ");
    }

    size_t cchSnippetLen = KStringLength(pwszMeaningfulText);

    TChar* pwszText = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, (static_cast<SIZE_T>(ulTextLenChars) + 1) * sizeof(TChar), TEST_POOL__TAG));
    
    if (!pwszText) [[unlikely]]
    {
        return FALSE;
    }

    for (ULONG i = 0; i < ulTextLenChars; i++)
    {
        pwszText[i] = pwszMeaningfulText[i % cchSnippetLen];
    }
    
    pwszText[ulTextLenChars] = static_cast<TChar>('\0');

    TChar* pStrArray = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, static_cast<SIZE_T>(ulPatternCount) * 128 * sizeof(TChar), TEST_POOL__TAG));
    
    if (!pStrArray) [[unlikely]]
    {
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    for (ULONG i = 0; i < ulPatternCount; i++)
    {
        if (bExpectMatch)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(&pStrArray[i * 128]), 128, "*allocation*%u*", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(&pStrArray[i * 128]), 128, L"*allocation*%u*", static_cast<unsigned int>(i));
            }
        }
        else
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(&pStrArray[i * 128]), 128, "*random*notfound*%u*", static_cast<unsigned int>(i));
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(&pStrArray[i * 128]), 128, L"*random*notfound*%u*", static_cast<unsigned int>(i));
            }
        }
        
        // Ensure at least one known good pattern matches if we expect a match
        if (bExpectMatch && i == ulPatternCount - 1)
        {
            if constexpr (sizeof(TChar) == 1)
            {
                sprintf_s(reinterpret_cast<char*>(&pStrArray[i * 128]), 128, "*allocation*");
            }
            else
            {
                swprintf_s(reinterpret_cast<wchar_t*>(&pStrArray[i * 128]), 128, L"*allocation*");
            }
        }

        PatternContext ctx = { i };
        
        if (!NT_SUCCESS(Nfa.AddPattern(&pStrArray[i * 128], KStringLength(&pStrArray[i * 128]), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx))) [[unlikely]]
        {
            ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
            ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
            return FALSE;
        }
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    LOG_INFO("\n[NFA_TEST] --- Perf Test [%s]: %s [%u Patterns | Text: %u Chars] ---\n", pszCharType, pszTestName, ulPatternCount, ulTextLenChars);
    LOG_INFO("[NFA_TEST]      %s\n", pszDescription);
    LOG_INFO("[NFA_TEST]      Pattern Type: %s\n", pszPatternType);
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");
    LOG_INFO("[NFA_TEST]      %-18s | %-15s | %-15s | %-15s\n", "Engine Mode", "Throughput", "Ops/sec", "Latency/Op");
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");

    LARGE_INTEGER liFreq;
    KeQueryPerformanceCounter(&liFreq);

    auto RunMode = [&](BOOLEAN bMatchFirst, const char* label) -> BOOLEAN
    {
        if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
        {
            return FALSE;
        }

        PerfWorkerCtx<TChar> Ctx = { 0 };
        Ctx.pNfa                 = &Nfa;
        Ctx.pText                = pwszText;
        Ctx.cchTextLen           = ulTextLenChars;
        Ctx.liFreq               = liFreq;
        Ctx.llTotalOpsPerSec     = 0;
        Ctx.bUseMatchFirst       = bMatchFirst;
        Ctx.bExpectMatch         = bExpectMatch;
        Ctx.lTestFailed          = 0;

        TEST_THREAD_MANAGER* pMgr = 
            static_cast<TEST_THREAD_MANAGER*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, sizeof(TEST_THREAD_MANAGER), TEST_POOL__TAG));
        
        if (pMgr) [[likely]]
        {
            StartThreads(pMgr, 1, PerfWorkerT<TChar>, &Ctx);
                         
            LARGE_INTEGER liDelay;
            liDelay.QuadPart = -2000000LL; 
            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);

            WriteRelease(&Ctx.lStartFlag, 1);
            
            // Wait for threads to clean up
            StopAndWaitThreads(pMgr, 2);
            
            if (ReadAcquire(&Ctx.lTestFailed) != 0) [[unlikely]]
            {
                LOG_ERR("[NFA_TEST]      %-18s | [!] FAILED: Unexpected Result (Expected %d, Actual %ld)\n", label, bExpectMatch, ReadNoFence(&Ctx.lActualResult));
                WriteRelease(&g_lAbortTests, 1);
                
                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return FALSE;
            }
            else
            {
                UINT64 opsPerSec = static_cast<UINT64>(ReadAcquire64(&Ctx.llTotalOpsPerSec));
                
                UINT64 totalPs = opsPerSec > 0 ? ((1000000000000ULL * (pMgr->ulThreadCount > 0 ? pMgr->ulThreadCount : 1)) / opsPerSec) : 0;
                UINT64 usWhole = totalPs / 1000000;
                UINT64 usFrac  = (totalPs % 1000000) / 100;

                // Only calculate and display MB/s if the engine was forced to process the entire buffer.
                // In this specific prose test, *allocation* matches within the first ~120 characters 
                // of the 100KB buffer, meaning MatchFirst legitimately exits early. 
                // Calculating MB/s using the full 100KB length would artificially inflate throughput to ~700 GB/s.
                if (!bMatchFirst || !bExpectMatch)
                {
                    UINT64 bytesPerOp  = static_cast<UINT64>(Ctx.cchTextLen) * sizeof(TChar);
                    UINT64 bytesPerSec = opsPerSec * bytesPerOp;
                    UINT64 mbWhole     = bytesPerSec / (1024ULL * 1024ULL);

                    if (mbWhole < 1)
                    {
                        UINT64 mbFrac = ((bytesPerSec % (1024ULL * 1024ULL)) * 100ULL) / (1024ULL * 1024ULL);
                        LOG_INFO("[NFA_TEST]      %-18s | %7llu.%02llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, mbFrac, opsPerSec, usWhole, usFrac);
                    }
                    else
                    {
                        LOG_INFO("[NFA_TEST]      %-18s | %10llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, opsPerSec, usWhole, usFrac);
                    }
                }
                else
                {
                    LOG_INFO("[NFA_TEST]      %-18s | %15s | %15llu | %9llu.%04llu us\n", label, "N/A (EarlyExit)", opsPerSec, usWhole, usFrac);
                }

                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return TRUE;
            }
        }
        
        return FALSE;
    };

    if (RunMode(FALSE, "MatchAll")) [[likely]]
    {
        RunMode(TRUE, "MatchFirst");
    }
    else
    {
        bTierSuccess = FALSE;
    }

    ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
    ExFreePoolWithTag(pwszText, TEST_POOL__TAG);

    return bTierSuccess;
}

// -------------------------------------------------------------------------------------
// Test Function: RunExactStringPerformanceTier
// Benchmarks purely exact string matching performance with no wildcards.
// -------------------------------------------------------------------------------------
template <typename TChar>
BOOLEAN RunExactStringPerformanceTier(_In_ ULONG       ulPatternCount,
                                      _In_ ULONG       ulTextLenChars,
                                      _In_ BOOLEAN     bExpectMatch,
                                      _In_ BOOLEAN     bCaseInsensitive,
                                      _In_ const char* pszTestName,
                                      _In_ const char* pszDescription,
                                      _In_ const char* pszPatternType,
                                      _In_ const char* pszCharType)
{
    PAGED_CODE();

    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        return FALSE;
    }

    KmWildcardNfa<PatternContext, TChar> Nfa(bCaseInsensitive);
    BOOLEAN                              bTierSuccess = TRUE;

    const SIZE_T cbTextMap = static_cast<SIZE_T>(ulTextLenChars) * sizeof(TChar) + sizeof(TChar);
    TChar*       pwszText  = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, cbTextMap, TEST_POOL__TAG));

    if (!pwszText) [[unlikely]]
    {
        return FALSE;
    }

    for (ULONG i = 0; i < ulTextLenChars; i++)
    {
        pwszText[i] = static_cast<TChar>('A' + (i % 26));
    }

    pwszText[ulTextLenChars] = static_cast<TChar>('\0');

    const SIZE_T cbPatternArray = static_cast<SIZE_T>(ulPatternCount) * cbTextMap;
    TChar*       pStrArray      = static_cast<TChar*>(ExAllocatePool2(POOL_FLAG_PAGED, cbPatternArray, TEST_POOL__TAG));

    if (!pStrArray) [[unlikely]]
    {
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    for (ULONG i = 0; i < ulPatternCount; i++)
    {
        TChar* pCurPat = &pStrArray[i * (ulTextLenChars + 1)];

        memcpy(pCurPat, pwszText, cbTextMap);

        if (!bExpectMatch || i != (ulPatternCount - 1))
        {
            if (ulTextLenChars > 0)
            {
                pCurPat[ulTextLenChars / 2] = static_cast<TChar>('X');
            }
        }

        PatternContext ctx = { i };

        if (!NT_SUCCESS(Nfa.AddPattern(pCurPat, KStringLength(pCurPat), KmWildcardNfa<PatternContext, TChar>::WILD_CARD_SCOPE::SCOPE_DEFAULT, ctx))) [[unlikely]]
        {
            ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
            ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
            return FALSE;
        }
    }

    if (!NT_SUCCESS(Nfa.Compile())) [[unlikely]]
    {
        ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
        ExFreePoolWithTag(pwszText, TEST_POOL__TAG);
        return FALSE;
    }

    LOG_INFO("\n[NFA_TEST] --- Perf Test [%s]: %s [%u Patterns | Text: %u Chars] ---\n", pszCharType, pszTestName, ulPatternCount, ulTextLenChars);
    LOG_INFO("[NFA_TEST]      %s\n", pszDescription);
    LOG_INFO("[NFA_TEST]      Pattern Type: %s\n", pszPatternType);
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");
    LOG_INFO("[NFA_TEST]      %-18s | %-15s | %-15s | %-15s\n", "Engine Mode", "Throughput", "Ops/sec", "Latency/Op");
    LOG_INFO("[NFA_TEST]      ------------------------------------------------------------------------------------\n");

    LARGE_INTEGER liFreq;
    KeQueryPerformanceCounter(&liFreq);

    auto RunMode = [&](BOOLEAN bMatchFirst, const char* label) -> BOOLEAN
    {
        if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
        {
            return FALSE;
        }

        PerfWorkerCtx<TChar> Ctx = { 0 };
        Ctx.pNfa                 = &Nfa;
        Ctx.pText                = pwszText;
        Ctx.cchTextLen           = ulTextLenChars;
        Ctx.liFreq               = liFreq;
        Ctx.llTotalOpsPerSec     = 0;
        Ctx.bUseMatchFirst       = bMatchFirst;
        Ctx.bExpectMatch         = bExpectMatch;
        Ctx.lTestFailed          = 0;

        TEST_THREAD_MANAGER* pMgr = 
            static_cast<TEST_THREAD_MANAGER*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, sizeof(TEST_THREAD_MANAGER), TEST_POOL__TAG));

        if (pMgr) [[likely]]
        {
            StartThreads(pMgr, 1, PerfWorkerT<TChar>, &Ctx);

            LARGE_INTEGER liDelay;
            liDelay.QuadPart = -2000000LL;
            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);

            WriteRelease(&Ctx.lStartFlag, 1);
            
            // Wait for threads to clean up
            StopAndWaitThreads(pMgr, 2);            

            if (ReadAcquire(&Ctx.lTestFailed) != 0) [[unlikely]]
            {
                LOG_ERR("[NFA_TEST]      %-18s | [!] FAILED: Unexpected Result (Expected %d, Actual %ld)\n", label, bExpectMatch, ReadNoFence(&Ctx.lActualResult));
                WriteRelease(&g_lAbortTests, 1);

                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return FALSE;
            }
            else
            {
                UINT64 opsPerSec = static_cast<UINT64>(ReadAcquire64(&Ctx.llTotalOpsPerSec));
                
                UINT64 totalPs = opsPerSec > 0 ? ((1000000000000ULL * (pMgr->ulThreadCount > 0 ? pMgr->ulThreadCount : 1)) / opsPerSec) : 0;
                UINT64 usWhole = totalPs / 1000000;
                UINT64 usFrac  = (totalPs % 1000000) / 100;

                // Test design ensures the entire string is scanned before the match succeeds or fails.
                UINT64 bytesPerOp  = static_cast<UINT64>(Ctx.cchTextLen) * sizeof(TChar);
                UINT64 bytesPerSec = opsPerSec * bytesPerOp;
                UINT64 mbWhole     = bytesPerSec / (1024ULL * 1024ULL);

                if (mbWhole < 1)
                {
                    UINT64 mbFrac = ((bytesPerSec % (1024ULL * 1024ULL)) * 100ULL) / (1024ULL * 1024ULL);
                    LOG_INFO("[NFA_TEST]      %-18s | %7llu.%02llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, mbFrac, opsPerSec, usWhole, usFrac);
                }
                else
                {
                    LOG_INFO("[NFA_TEST]      %-18s | %10llu MB/s | %15llu | %9llu.%04llu us\n", label, mbWhole, opsPerSec, usWhole, usFrac);
                }

                ExFreePoolWithTag(pMgr, TEST_POOL__TAG);
                return TRUE;
            }
        }

        return FALSE;
    };

    if (RunMode(FALSE, "MatchAll")) [[likely]]
    {
        RunMode(TRUE, "MatchFirst");
    }
    else
    {
        bTierSuccess = FALSE;
    }

    ExFreePoolWithTag(pStrArray, TEST_POOL__TAG);
    ExFreePoolWithTag(pwszText, TEST_POOL__TAG);

    return bTierSuccess;
}

// -------------------------------------------------------------------------------------
// Unified helper wrapping boilerplate execution of performance tiers for any text base.
// -------------------------------------------------------------------------------------
template <typename TChar>
VOID ExecutePerformanceTiers(_Inout_ ULONG&      ulPassed,
                             _Inout_ ULONG&      ulFailed,
                             _In_    const char* pszCharType)
{
    auto CheckTest = [&](BOOLEAN bResult)
    {
        // If the global abort flag is set due to a catastrophic perf test logic failure, exit the block.
        if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
        {
            return;
        }

        if (bResult) [[likely]]
        {
            ulPassed++;
        }
        else
        {
            ulFailed++;
        }
    };

    CheckTest(RunPerformanceTier<TChar>(10, 512, TRUE, TRUE, TRUE, "Small Pathological", "Evaluates throughput against a saturated set of short pathological wildcard rules.", "Pathological (*A%dB*)", pszCharType));

    CheckTest(RunPerformanceTier<TChar>(100, 10240, TRUE, TRUE, TRUE, "Medium Pathological", "Evaluates throughput against a saturated set of medium pathological wildcard rules.", "Pathological (*A%dB*)", pszCharType));

    CheckTest(RunPerformanceTier<TChar>(1000, 102400, TRUE, TRUE, TRUE, "Large Pathological", "Evaluates throughput against a saturated set of long pathological wildcard rules.", "Pathological (*A%dB*)", pszCharType));

    CheckTest(RunPerformanceTier<TChar>(100, 10240, TRUE, FALSE, TRUE, "Negative Mismatch", "Tests engine behavior when no patterns match the target string.", "Pathological (*A%dB*)", pszCharType));

    CheckTest(RunRealisticPathPerformanceTier<TChar>(10, TRUE, TRUE, "Realistic Paths", "Benchmarks against a small set of typical kernel file paths.", "File Paths (\\Device\\...)", pszCharType));

    CheckTest(RunRealisticPathPerformanceTier<TChar>(100, TRUE, TRUE, "Realistic Paths", "Benchmarks against a medium set of typical kernel file paths.", "File Paths (\\Device\\...)", pszCharType));

    CheckTest(RunRealisticPathPerformanceTier<TChar>(1000, TRUE, TRUE, "Realistic Paths", "Benchmarks against a massive set of typical kernel file paths.", "File Paths (\\Device\\...)", pszCharType));

    CheckTest(RunRealisticPathPerformanceTier<TChar>(10000, TRUE, TRUE, "Realistic Paths", "Benchmarks against a massive set of typical kernel file paths.", "File Paths (\\Device\\...)", pszCharType));

    CheckTest(RunRealisticPathPerformanceTier<TChar>(1000, FALSE, TRUE, "Negative Paths", "Tests engine behavior when no paths match the target.", "File Paths (\\Device\\...)", pszCharType));

    CheckTest(RunMeaningfulTextPerformanceTier<TChar>(100, 1024, TRUE, TRUE, "Meaningful Text", "Simulates 1KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)", pszCharType));

    CheckTest(RunMeaningfulTextPerformanceTier<TChar>(100, 10240, TRUE, TRUE, "Meaningful Text", "Simulates 10KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)", pszCharType));

    CheckTest(RunMeaningfulTextPerformanceTier<TChar>(100, 102400, TRUE, TRUE, "Meaningful Text", "Simulates 100KB real-world prose with multi-wildcard boundaries.", "Prose Boundaries (*xxx*)", pszCharType));

    CheckTest(RunMeaningfulTextPerformanceTier<TChar>(100, 102400, FALSE, TRUE, "Meaningful Negative", "Tests mismatch behavior on 100KB real-world prose context.", "Prose Boundaries (*xxx*)", pszCharType));

    CheckTest(RunMeaningfulTextPerformanceTier<TChar>(100, 102400, TRUE, FALSE, "Case-Sensitive Prose", "Simulates 100KB prose utilizing raw character evaluations.", "Prose Boundaries (*xxx*)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(10, 128, TRUE, TRUE, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(100, 128, TRUE, TRUE, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(1000, 128, TRUE, TRUE, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(10000, 128, TRUE, TRUE, "Exact String", "Simulates 128-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));
    
    CheckTest(RunExactStringPerformanceTier<TChar>(100, 128, FALSE, TRUE, "Exact String Negative", "Tests mismatch behavior on 128-char exact string context.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(10, 256, TRUE, TRUE, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(100, 256, TRUE, TRUE, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));

    CheckTest(RunExactStringPerformanceTier<TChar>(1000, 256, TRUE, TRUE, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));
    
    CheckTest(RunExactStringPerformanceTier<TChar>(10000, 256, TRUE, TRUE, "Exact String", "Simulates 256-char text with exact string matching.", "Exact Literal (No Wildcards)", pszCharType));
    
    CheckTest(RunExactStringPerformanceTier<TChar>(100, 256, FALSE, TRUE, "Exact String Negative", "Tests mismatch behavior on 256-char exact string context.", "Exact Literal (No Wildcards)", pszCharType));
}

// -------------------------------------------------------------------------------------
// Orchestrates all performance test tiers for both char strings.
// -------------------------------------------------------------------------------------
VOID 
RunAllPerformanceTests()
{
    PAGED_CODE();

    LOG_INFO("\n[NFA_TEST] =========================================================================================\n");
    LOG_INFO("[NFA_TEST] --- Performance Tests (Throughput) ---\n");
    LOG_INFO("[NFA_TEST] =========================================================================================\n");

    ULONG ulPassed = 0;
    ULONG ulFailed = 0;

    ExecutePerformanceTiers<WCHAR>(ulPassed, ulFailed, "WCHAR");
    ExecutePerformanceTiers<CHAR>(ulPassed, ulFailed, "CHAR");

    if (ReadAcquire(&g_lAbortTests) != 0) [[unlikely]]
    {
        LOG_ERR("\n[NFA_TEST] [!] PERFORMANCE SUITE ABORTED DUE TO CRITICAL THREAD LOGIC FAILURE\n");
    }
    else if (ulFailed == 0) [[likely]]
    {
        LOG_INFO("\n[NFA_TEST] [+] ALL PERFORMANCE TESTS SUCCEEDED (%u passed)\n", ulPassed);
    }
    else
    {
        LOG_ERR("\n[NFA_TEST] [!] SOME PERFORMANCE TESTS FAILED (%u passed, %u failed)\n", ulPassed, ulFailed);
    }
}

// -------------------------------------------------------------------------------------
// Main routine that bootstraps all correctness checks and triggers performance evaluation.
// -------------------------------------------------------------------------------------
VOID RunTests(_In_opt_ PVOID pContext)
{
    PAGED_CODE();

    TEST_SUITE_CONFIG* pConfig = static_cast<TEST_SUITE_CONFIG*>(pContext);

    KeSetPriorityThread(KeGetCurrentThread(), 2); 
    
    LOG_INFO("\n[NFA_TEST] =========================================================================================\n");
    LOG_INFO("[NFA_TEST]          KMWILDCARDNFA UNIFIED TEST SUITE\n");
    LOG_INFO("[NFA_TEST] =========================================================================================\n");

    // Correctness checks are always prioritized before executing expensive bandwidth operations
    if (pConfig && pConfig->bRunCorrectness) [[likely]]
    {
        RunAllCorrectnessTests();
    }

    if (pConfig && pConfig->bRunPerformance) [[likely]]
    {
        RunAllPerformanceTests();
    }

    LOG_INFO("\n[NFA_TEST] --- All Tests Finished ---\n\n");
    
    FlushLogToFile();
    PsTerminateSystemThread(STATUS_SUCCESS);
}