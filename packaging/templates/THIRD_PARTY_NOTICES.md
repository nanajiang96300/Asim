# 第三方组件及授权

各组件原始许可证/源码版权声明随源码保留，不在本说明中重新授权。交付或商业部署前应由使用单位核对适用条款；本项目自身授权请向负责人确认。

| 组件 | 来源版本/说明 | 许可证位置 |
|---|---|---|
| Boost | 1.79.0 headers/program_options | vendor/include/boost/LICENSE_1_0.txt |
| robin-hood-hashing | 3.11.5 | vendor/licenses/robin-hood-hashing/ |
| nlohmann_json | 3.11.2 | vendor/licenses/nlohmann_json/ |
| spdlog + bundled fmt | 1.11.0 | vendor/spdlog/LICENSE及头部声明 |
| argparse | 2.9 | vendor/argparse/LICENSE |
| yaml-cpp | 0.7.0 | vendor/yaml-cpp/LICENSE |
| ONNX | 见SOURCE_PROVENANCE.json | extern/onnx/LICENSE |
| Protobuf | 3.18.1 | extern/protobuf/LICENSE |
| Ramulator2 | 工程现有源码 | extern/ramulator2/LICENSE |
| BookSim、Ramulator custom | 工程现有源码 | extern下对应源文件版权声明（原树无独立LICENSE） |
| NumPy | 2.5.0官方wheel及其包含的库 | vendor/wheels中wheel内dist-info/licenses |
| pip | 26.1.2官方wheel | vendor/wheels中wheel内dist-info/licenses |

不包含私有Conan账号设置、缓存元数据、系统库二进制、模型权重或研究数据。自动下载不在安装流程内；已提供Python wheel仅适配CPython3.12/Linux x86_64。
