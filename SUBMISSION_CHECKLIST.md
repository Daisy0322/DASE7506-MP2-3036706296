# MP2 提交清单（学号 3036706296）

截止时间：**2026-10-31 23:59（UTC+8）**

正式成绩：**83.72591006423983**（`report/test-score.json`，391/467）

## 已准备完成（不用再动）

- [x] `code/submission.py`、`code/selector.py`、`code/checks.py`：官方 `predict` 入口与推理实现
- [x] `code/train.py`：可复现训练（seed 0，复现 checkpoint 哈希）
- [x] `code/checkpoint/model_bundle/`：冻结 checkpoint（147,098 字节）
- [x] `report/REPORT.md`：报告（含 AI disclosure、消融、两阶段实验分析）
- [x] `report/test-score.json`、`report/validation-score.json`：官方评分输出
- [x] `experiments/two_stage/`：交叉验证选择的两阶段实验（测试 82.44%，不是提交成绩，报告中已说明）
- [x] 根目录 `README.md`：安装、复现命令、哈希
- [x] 本地 git 仓库已初始化、已提交、已打标签 `mp2-final-3036706296`

## 你必须亲自完成（约 10 分钟）

### 1. 在 GitHub 上建一个空的公开仓库

1. 登录 <https://github.com>，点右上角 `+` → `New repository`。
2. Repository name 填 `DASE7506-MP2-3036706296`。
3. 选 **Public**。
4. **不要**勾选 Add a README / .gitignore / license。
5. 点 `Create repository`。

### 2. 推送（在本目录打开终端，替换 `<你的GitHub用户名>`）

```bash
cd "/Users/xinyi/Desktop/course/7506_Advenced ML/DASE7506-MP2-main"
git remote add origin https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296.git
git push -u origin main
git push origin mp2-final-3036706296
```

弹出登录窗口就用浏览器登录 GitHub。如果终端要密码，要填 **Personal Access Token**
（GitHub → Settings → Developer settings → Personal access tokens → Generate new token，
勾选 `repo`），不能填 GitHub 登录密码。仓库约 105 MB，推送需要一两分钟。

### 3. 用无痕窗口确认链接能打开

- `https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296/tree/mp2-final-3036706296`
- `https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296/blob/mp2-final-3036706296/report/REPORT.md`
- `https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296/tree/mp2-final-3036706296/code/checkpoint/model_bundle`

### 4. 在课程 MP2 网站提交

打开 <https://xudongwu-0.github.io/courses/dase7506/mp2/>，填写：

| 字段 | 填写内容 |
|---|---|
| Student ID | `3036706296` |
| Score / accuracy | `83.72591006423983` |
| Code link | `https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296/tree/mp2-final-3036706296` |
| Checkpoint link | `https://github.com/<你的GitHub用户名>/DASE7506-MP2-3036706296/tree/mp2-final-3036706296/code/checkpoint/model_bundle` |

网站会生成一个 GitHub issue 页面：**必须点 "Submit on GitHub"，并在 GitHub 页面上真正点击创建 issue**。
只生成链接不算提交。

### 5. 最后核对

- [ ] 仓库是 Public，无痕窗口能打开上面三个链接
- [ ] 网站成绩是 `83.72591006423983`，学号 `3036706296`
- [ ] GitHub issue 已真正创建（页面上能看到 issue 编号）
- [ ] 截图保存网站结果和 issue 页面

`.venv/`、`code/runs/`、本地备份 ZIP 已被 `.gitignore` 排除，不会推上去。
