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
// User Mode Test Suite App for Wildcard Matching Engine
// -------------------------------------------------------------------------------------
#include "ntddk.h"
#include <stdint.h>
#include <intrin.h>

// -------------------------------------------------------------------------------------
// Logging Setup
// -------------------------------------------------------------------------------------
#define LOG_INFO(...)      do { printf(__VA_ARGS__); fflush(stdout); } while(0)
#define LOG_ERR(...)       do { printf(__VA_ARGS__); fflush(stdout); } while(0)

// -------------------------------------------------------------------------------------
// No-op log flush stub matching the kernel driver interface (user-mode output streams
// directly to stdout via LOG_INFO / LOG_ERR).
// -------------------------------------------------------------------------------------
inline VOID FlushLogToFile()
{
}

// -------------------------------------------------------------------------------------
// Globals
// -------------------------------------------------------------------------------------
volatile LONG g_lAbortTests            = 0;    // Atomic flag compatible with kernel abort checks (unused in UM)
PETHREAD      g_pMasterBenchmarkThread = NULL; // Emulated thread handle executing RunTests

#include "KmWildcardNfa.h"
#include "TestKmWildcardNfa.h"

TEST_SUITE_CONFIG g_TestConfig = { TRUE, TRUE }; // Suite configuration enabling correctness and performance tests

// -------------------------------------------------------------------------------------
// User-mode entry point that spawns the emulated system thread running RunTests and
// waits for suite completion before exiting.
// -------------------------------------------------------------------------------------
int main()
{    
    LOG_INFO("[NFA_TEST] Performance tests running with UM Threading / Memory Emulation.\n\n");

    HANDLE   hThread;
    NTSTATUS ntStatus = PsCreateSystemThread(&hThread, 
                                             0, 
                                             NULL, 
                                             NULL, 
                                             NULL, 
                                             (PKSTART_ROUTINE)RunTests,
                                             &g_TestConfig);

    if (NT_SUCCESS(ntStatus))
    {
        ntStatus = ObReferenceObjectByHandle(hThread, 0, NULL, 0, (PVOID*)&g_pMasterBenchmarkThread, NULL);
        ZwClose(hThread);
        
        if (g_pMasterBenchmarkThread != NULL)
        {
            KeWaitForSingleObject(g_pMasterBenchmarkThread, Executive, KernelMode, FALSE, NULL);
            ObDereferenceObject(g_pMasterBenchmarkThread);
            g_pMasterBenchmarkThread = NULL;
        }
    }

    LOG_INFO("[NFA_TEST] *** User-Mode Test Application Exited Safely.\n");
    return 0;
}