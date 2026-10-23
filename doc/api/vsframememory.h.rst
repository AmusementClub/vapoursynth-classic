VSFrameMemory.h
===============

Table of contents
#################

Introduction_

Structs_
   VSFrameMemoryAPI_

Functions_
   getVSFrameMemoryAPI_


Introduction
############

A vapoursynth-classic extension for plugins that move frames to and from a
device (CUDA, Vulkan, ...).

Frame planes live in blocks of the core's frame pool. A freed block is kept
and reused for later frames, and is returned to the system only when the pool
shrinks below its limit (see *setMaxCacheSize*) or is destroyed. Blocks of
64 KiB and more start and end on page boundaries.

A plugin can therefore register a block with a device API once
(``cudaHostRegister``, ``VK_EXT_external_memory_host``) and then let the
device copy straight from source frames and into output frames, instead of
copying through its own pinned staging buffers. The release hook tells the
plugin when a block is about to be returned to the system, so that it can
unregister it first.

Registering memory does not change it for anyone else: the pages stay normal
readable and writable memory and are only kept resident. Filters that do not
use this extension need no changes.

The extension is not part of *VSAPI*. Look up *getVSFrameMemoryAPI* in the
library that provides the *VSAPI* in use, for example with ``dladdr()`` on a
*VSAPI* function pointer followed by ``dlopen(..., RTLD_NOLOAD)`` and
``dlsym()``, or ``GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, ...)``
and ``GetProcAddress()`` on Windows. Treat a missing symbol as "not available".


Structs
#######

VSFrameMemoryAPI
----------------

   .. c:member:: int version

      *VS_FRAME_MEMORY_API_VERSION* of the library.

   .. c:member:: size_t pageSize

      The system page size. Blocks returned by *getPlaneBlock* start and end
      on multiples of it.

   ----------

   .. _getPlaneBlock:

   int getPlaneBlock(const VSFrame \*f, int plane, void \*\*base, size_t \*size)

      Returns in *base* and *size* the pool block holding *plane* of *f*,
      which includes the plane's data. The block stays allocated at least as
      long as *f*, and may hold planes of later frames once *f* is freed.

      Returns 0 (and leaves *base* and *size* alone) when the plane does not
      live in a page-aligned pool block, for example a plane smaller than
      64 KiB.

   ----------

   .. _addReleaseHook:

   int addReleaseHook(VSCore \*core, VSFrameMemoryReleaseHook hook, void \*userData)

      Calls ``hook(userData, base, size)`` for every page-aligned block of
      *core*'s frame pool right before it is returned to the system: when the
      pool shrinks, and when the pool itself is destroyed, which can happen
      after the core is freed if frames outlive it.

      The hook runs on the thread that frees the block, with no core lock
      held. Hooks never run concurrently. A hook must not call any
      VapourSynth or *VSFrameMemoryAPI* function.

      Returns an id greater than 0, or 0 on failure.

   ----------

   .. _removeReleaseHook:

   void removeReleaseHook(VSCore \*core, int id)

      Removes a hook. When this returns, the hook is not running and is never
      called again. Do not call it while holding a lock the hook takes.

      Remove the hook before the plugin's last filter instance on the core is
      freed: the pool can outlive both the core and the plugin.


Functions
#########

getVSFrameMemoryAPI
-------------------

.. c:function:: const VSFrameMemoryAPI *getVSFrameMemoryAPI(int version)

   Returns the extension for *version* (1 up to
   *VS_FRAME_MEMORY_API_VERSION*), or NULL if that version is not supported.
