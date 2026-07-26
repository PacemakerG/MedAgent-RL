# 原始中文医疗数据集

该目录用于在本地保存 DoctorAgent-RL 论文涉及的三个中文医疗数据源：

```text
data/raw_sources/
├── IMCS21/
├── MedDG/
└── CHIP-MDCFNPC/
```

## 下载

在仓库根目录执行：

```bash
bash self_scripts/download_raw_datasets.sh
```

脚本会自动：

- 通过 GitHub SSH 下载或更新 IMCS-21。
- 通过 GitHub SSH 下载或更新 MedDG。
- 检查 CHIP-MDCFNPC 的 ZIP 是否已经放入本地目录，并在存在时自动解压。

CHIP-MDCFNPC 的官方天池下载入口需要登录，需先手动下载：

```text
https://tianchi.aliyun.com/dataset/95414
```

下载后将文件放到：

```text
data/raw_sources/CHIP-MDCFNPC/CHIP-MDCFNPC.zip
```

然后重新运行下载脚本。

## 数据来源

- IMCS-21：`lemuria-wchen/imcs21`
- MedDG：`lwgkzl/MedDG`
- CHIP-MDCFNPC：天池 CBLUE 数据集

## 为什么不提交原始数据

原始第三方数据保留在本地，不提交到本仓库，原因包括：

- 保留原数据来源和版本信息。
- 避免重复分发第三方数据。
- 避免 Git 仓库体积持续增大。
- 后续清洗脚本可以从固定的本地目录读取。

后续由本项目生成的小规模清洗样本、统一格式样本和 CoT SFT 样本，可以提交到 `data/processed_samples/`。
