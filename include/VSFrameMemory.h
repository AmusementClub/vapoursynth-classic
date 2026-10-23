/*
* Copyright (c) 2026 VapourSynth Classic contributors
*
* This file is part of VapourSynth.
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

/*
 * vapoursynth-classic extension: frame memory blocks and their release.
 *
 * Frame planes live in blocks of the core's frame pool, which are reused for
 * later frames and returned to the system only when the pool shrinks. A
 * plugin that moves frames to and from a device (CUDA, Vulkan) can register a
 * block with the device API once (cudaHostRegister,
 * VK_EXT_external_memory_host) and then copy straight from source frames and
 * into output frames, without staging copies. The release hook tells it when
 * a block is about to go back to the system, so that it can unregister it
 * first; without it a registration could outlive its memory.
 *
 * The extension is not part of VSAPI: find it with getVSFrameMemoryAPI() in
 * the library that provides the VSAPI (for example by dladdr() / GetModuleHandleEx()
 * on a VSAPI function pointer), and treat its absence as "not available".
 */

#ifndef VSFRAMEMEMORY_H
#define VSFRAMEMEMORY_H

#include "VapourSynth4.h"

#define VS_FRAME_MEMORY_API_VERSION 1

/* Called right before the block [base, base + size) is returned to the system. */
typedef void (VS_CC *VSFrameMemoryReleaseHook)(void *userData, void *base, size_t size);

typedef struct VSFrameMemoryAPI {
    int version; /* VS_FRAME_MEMORY_API_VERSION */
    size_t pageSize; /* blocks returned by getPlaneBlock start and end on multiples of it */

    /* The pool block holding plane `plane` of f, which includes the plane's
     * data. base and size are multiples of pageSize. The block stays allocated
     * at least as long as f and may hold planes of later frames once f is
     * freed. Returns 0 (and leaves base and size alone) when the plane does
     * not live in a page-aligned pool block, for example a small plane. */
    int (VS_CC *getPlaneBlock)(const VSFrame *f, int plane, void **base, size_t *size) VS_NOEXCEPT;

    /* Calls hook(userData, base, size) for every block of core's frame pool
     * right before it is returned to the system: when the pool shrinks, and
     * when the pool itself is destroyed (which may happen after the core is
     * freed, if frames outlive it). The hook runs on the freeing thread with
     * no core lock held, hooks never run concurrently, and the hook must not
     * call any VapourSynth or VSFrameMemoryAPI function. Returns an id > 0, or
     * 0 on failure. */
    int (VS_CC *addReleaseHook)(VSCore *core, VSFrameMemoryReleaseHook hook, void *userData) VS_NOEXCEPT;

    /* Removes a hook. When this returns the hook is not running and is never
     * called again. Do not call it while holding a lock the hook takes.
     * Remove the hook before the plugin's last filter instance on the core is
     * freed: the pool can outlive the core and the plugin. */
    void (VS_CC *removeReleaseHook)(VSCore *core, int id) VS_NOEXCEPT;
} VSFrameMemoryAPI;

VS_API(const VSFrameMemoryAPI *) getVSFrameMemoryAPI(int version) VS_NOEXCEPT;

#endif /* VSFRAMEMEMORY_H */
