# Git 提交与推送规则（模板�?
> 一条总规则：**改完一个有意义的功�?修复，就立刻提交并推�?*——把 GitHub 当备份，不要攒着�?
## 每次推送的固定三步（日常模板）

```powershell
cd E:\oprncode\c-former          # 1. 进项目目�?git add -A                        # 2. 暂存所有改�?git commit -m "一句话说明改了什�?   # 3. 提交
git push                          # 4. 推到 GitHub（已记住地址，不用写全）
```

## 提交信息怎么写（模板�?
中文，一句话，说清「改了什么」：

```powershell
git commit -m "修复：区域轴近义别名混淆"
git commit -m "新增：V6.2 观测点端到端问答"
git commit -m "chore：更新依赖版�?
```

## 什么时候打版本标签（规则）

- **只有里程�?正式节点才打 tag**，日常调试不打：

```powershell
git tag v0.6.2 -m "V6.2 观测点端到端"
git push origin v0.6.2
```

- 版本号规则：**测试版用 0.x.y，正式版才用 1.0.0**�?
## 不要提交的东�?
- `artifacts/`（模型检查点、大数据）已经写�?`.gitignore`，会自动忽略，不用管它�?
## 常见情况速查

| 情况 | 命令 |
|---|---|
| 日常小改�?| `add -A` �?`commit` �?`push` |
| 看改了哪些文�?| `git status` |
| 看具体改动内�?| `git diff` |
| 看提交历�?| `git log --oneline` |
| 撤销某个文件的改�?| `git checkout -- 文件名` |
| 推送报�?| 把报错原文贴出来，别乱敲命令 |

## 一个完整的例子（以后照着做）

```powershell
cd E:\oprncode\c-former
git add -A
git commit -m "修复：ANN 512K 显存溢出"
git push
```
