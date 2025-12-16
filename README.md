# Requirements
## clang
## g++
## CMake
## SDL3
## Vulkan


# strategia alokatora:
Warstwa 0 (OS): System operacyjny daje RAZ wielki blok (np. 1GB).

Warstwa 1 (Master Allocator): Zarządza tym 1GB. Zazwyczaj LinearArena (Stack).

Warstwa 2 (System Allocators): Fizyka prosi Mastera: "Daj mi 50MB". Renderer prosi: "Daj mi 200MB".

Warstwa 3 (Local Allocators): Wewnątrz tych 50MB Fizyki tworzony jest PoolAllocator.
