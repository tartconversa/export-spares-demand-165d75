# 海外储能售后备件的隐性需求与保障方案

现有内容是合成业务资料及通用资料检查器，以及在其上构建的备件保障分析应用
（`spares/` 包）。字段、单位和业务记录的含义见 docs/资料口径.md，
应用使用方式与口径解释见 docs/应用使用说明.md，完整性目录为 data/catalog.json。
不得把文件自检成功解释成业务结论成立。

## 环境

Python 3.11.2；分析依赖已在 requirements.txt 固定版本。

    python3 -m pip install -r requirements.txt
    python3 -m sourcepack

检查入口读取资料目录，核对文件摘要、字段类型、记录键、引用与行数，不进行估计、推荐、批准或结算。新增业务资料可使用相同字段契约，目录摘要只对应当前保存的输入文件。

## 已有检查与语法构建

    python3 -m unittest discover -s tests -v
    python3 -m compileall -q sourcepack tests

现有测试只覆盖资料容器和输入诊断。备件保障应用的业务口径测试见
tests/test_business.py（27 项），运行：

    python3 -I -m unittest discover -s tests -p "test_business.py" -v
    python3 -m compileall -q spares tests

应用入口（隔离模式，避免工作目录植入模块）：

    python3 -I spares/__main__.py --data data --sims 500 forecast
    python3 -I spares/__main__.py --data data --sims 500 plan --scenario normal

完整季度演练（预测→选优→落实→模拟实际→回看→HTML 报告）见
docs/应用使用说明.md。运行资料检查不需要访问外部业务系统。
