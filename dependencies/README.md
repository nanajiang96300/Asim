# 固定依赖

lock.json锁定依赖分支的完整commit、每项大小/SHA和源码manifest SHA。源码bundle是已验收专家包相同第三方源码/生成ONNX C++，包含许可证；wheel为相同官方pip26.1.2/NumPy2.5.0 Linux x86_64 Python3.12制品。无Asim二进制或数据。主分支不提交展开依赖。

首次运行python3 scripts/bootstrap.py会下载固定URL至.deps/cache，安全解压并逐文件检查，构建时不联网。不读取Conan或用户API配置。企业离线使用--cache /媒体/cache --offline；缓存须包含lock列出的三个原名文件，内容不匹配直接拒绝、不覆盖。可用--download-only在联网机器准备缓存。CMake独立配置前也会执行只读依赖校验。

依赖升级必须新commit/新锁/新验收，不改旧锁对应制品。--deps-dir与--build-dir可指定独立路径；同一可写目录不并发bootstrap。默认串行构建；ASIM_USE_PCH=ON仅为加速源代码编译，可手动配置为OFF，不改变模拟指令。
