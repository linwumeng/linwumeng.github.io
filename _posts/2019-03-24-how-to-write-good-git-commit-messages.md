---
title: "How to write good git commit messages"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - git
  - best practice
typora-root-url: ..\
---
![](https://imgs.xkcd.com/comics/git_commit_2x.png)
It's totally true that the commit messages of a repo become less and less informative as project goes. The article, [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/), gives good guidance to address the issue.

The seven suggestions are,
1. Use a blank line to separate title.
1. Don't use period at the end of the title.
1. Capitalize the title.
1. Use imperative mood to write the title.
1. Keep the title in 50 words.
1. Wrap the body in 70 words a line.
1. Use the body to explain what and why instead of how.

Further suggestions are,
1. Prefer to small checking with [WIP] in title.
1. Think of your reviewer by review your pull request first to ensure it's reviewable.
1. Use `git rebase -i HASH` to revise and squarsh commits for a releasable feature.