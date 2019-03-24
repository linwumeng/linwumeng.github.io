---
title: "How to use vscode as git editor"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - git
  - best practice
typora-root-url: ..\
---

Stackoverflow gives [Instructions](https://stackoverflow.com/questions/30024353/how-to-use-visual-studio-code-as-default-editor-for-git/36644561#36644561) to make vscode as git editor.

1. Ensure `Code` is on shell path
1. Set `Code` as the git editor
> `git config --global core.editor "code --wait"`
3. Use `Code` to edit configurations
> `git config --global -e`

![](https://i.stack.imgur.com/gygTe.png)

4. Add configurations to use `Code` as default diff tool
```
[diff]
    tool = default-difftool
[difftool "default-difftool"]
    cmd = code --wait --diff $LOCAL $REMOTE
```