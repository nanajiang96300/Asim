# RW probe：框架事件验收微程序

## 1. 文件

src/semantic/RwProbeModel.h/.cc；mode rw_probe_test。仅用于受信框架正负例，非通信优化候选。

## 2. 公式推导

输入A/B为16×16 FP64。先C=A@B，再D=A/A。serialized用依赖C的barrier后令D覆盖C，最终STORE C应为全1矩阵；independent写独立D，STORE C应为A@B，允许并行。missing_barrier将D写C但删依赖barrier，Vector先结束、Cube后结束造成真实WAW；overwrite_input在Cube保护读窗口内写A，保守WAR模型拒绝。

## 3. 布局与指令

A/B置SPAD，C/D置ACCUM，sync单独64byte。真实LOAD、GEMM、DIV、barrier、STORE；每个source-view对应同一Instruction，正负例只改变声明的同步/目标。所有外部输入显式提供，缺LOAD不补数值。

## 4. 构建/审查/验收

设计先于编码；隔离build-j1；真实Asim正例结束且公共重放与矩阵参考一致，负例由真实读写流拒绝。独立审查Core observer与Python绑定器；大制品单独目录保存。FormulaLogger仅诊断。未知事件或模型不猜测。

DIV采用非零非均匀A；其32周期Vector区间与Cube实际计算区间重叠，正例不得只把CubeWait当Cube计算重叠。
