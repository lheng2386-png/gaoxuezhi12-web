# Windows 桌面独立应用打包说明

> 注意：当前软件著作权申请建议以“高血脂风险分层预警与干预推荐系统 Web版 V1.0”为主。本文件仅作为可选桌面打包说明保留，若本次只申请 Web 版，可不提交本文件，避免软件形态描述混杂。

## 项目结构

```
hyperlipidemia-risk-system/
├── README.md
├── 软著申请说明.md
├── Windows打包说明.md      # 本文件
├── backend/               # 后端API（完整核心算法）
├── frontend/              # Web前端版本
└── desktop/              # Windows桌面独立应用
    ├── gui.py             # 主GUI程序（PyQt5）
    └── requirements.txt   # 依赖
```

## 在Windows上打包步骤

### 1. 环境准备

在你的 Windows 电脑上：

```bash
# 安装Python 3.8+ （从 python.org 下载安装）
# 安装依赖
cd desktop
pip install -r requirements.txt
# 安装后端依赖也需要
cd ../backend
pip install -r requirements.txt
```

### 2. 测试运行

```bash
cd desktop
python gui.py
```

应该能看到GUI窗口打开，可以测试输入计算。

### 3. 打包为独立exe

```bash
# 在项目根目录
pyinstaller hyperlipidemia-risk-system.spec
```

打包完成后，exe文件在 `dist/高血脂风险预警系统.exe`

### 4. 分发

- 直接把 `dist/` 目录里的 `.exe` 发给别人就能用
- 或者把整个 `dist` 文件夹压缩分发

## 说明

- 这是**单文件独立桌面应用**，不需要安装，双击就能运行
- 所有三大核心算法都完整包含在exe里，100%按照论文实现
- GUI界面原生 Windows 运行，不需要浏览器，不需要启动后端服务

## 文件大小

打包后大约 50-100 MB（包含 Python 运行时），属于正常范围。
