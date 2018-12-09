---
layout: post
title: "How do I create this blog?"
description: ""
category: lessons 
tagline: "Supporting tagline"
tags: [intro, beginner, jekyll, tutorial]
excerpt_separator: <!--more--> 
---
{% include JB/setup %}

## Preface
I create this blog to record how I grow as my study. By study ways to create a blog, I choose the solution of publishing blog by Jekyll on Github. This solution wins for it is easy to build up and maintain in the future.

There are articles online to address how to use Github to host your pages with Jekyll to be a blog system. As the author of the [article](erjjones.github.io/blog/How-I-built-my-blog-in-one-day/) I followed, I don't invent a new way to do the same thing. I decide to write this article for purposes:
<!--more-->

&nbsp;&nbsp;&nbsp;&nbsp;1.   It's a test to be my first article to use the system.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;2.   I meet some problems on the way to create this blog.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I want to review the process here so that I and someone<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else could read it to get help.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;3.   I grow from this experience. I feel good!<br/>

Besides the [article](erjjones.github.io/blog/How-I-built-my-blog-in-one-day/), I get the blog running by following the tutorial [jekyllbootstrap](http://jekyllbootstrap.com) in fact. The tutorial can help setup your blog on GitHub in 3 minutes as its promise!

## GitHub
It's not supprise that the first step is to register a GitHub account since I want to use GitHub Pages to host the blog.

&nbsp;&nbsp;&nbsp;&nbsp;1.   Setup a free [GitHub](https://github.com/signup/free) account, and follow the [setup instructions](http://help.github.com/).<br/>
&nbsp;&nbsp;&nbsp;&nbsp;2.   Create a [GitHub Page](http://pages.github.com) with your username.

After these steps, I get an empty repository, linwumeng.github.io, to host the pages.

## Jekyll
[Jekyll](http://jekyllrb.com) is a static site generator. It's not a blog software, but it is fairly easy to generate static pages of your blog pages.
To use Jekyll, I need some initial code for Jekyll to generate pages. The [tutorial](http://jekyllbootstrap.com) states the steps in details.

Simply speaking, I need to clone a git repo to the local disk, then publish the site by push the files to my git repo. GitHub is able to recognize Jekyll folder layout and host the content. After the init push, I can access the init content by accessing http://linwumeng.github.io.

It's time to create my own content and publish them as blogs. The main cycle is to use Jekyll to create pages or posts and then push to the [repo](https://github.com/linwumeng/linwumeng.github.io).

#### Steps to publish content
&nbsp;&nbsp;&nbsp;&nbsp;1.   Clone a Jekyll repo.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Option I: Clone the repo [jekyll-bootstrap](https://github.com/plusjade/jekyll-bootstrap.git)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Option II: Clone the repo [erjjones](https://github.com/plusjade/jekyll-bootstrap.git)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Option III: Clone the repo [poole](https://github.com/plusjade/jekyll-bootstrap.git)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;2.   Push the code to my repo.<br/>

	git clone https://github.com/plusjade/jekyll-bootstrap.git USERNAME.github.com
	cd USERNAME.github.com
	git remote set-url origin git@github.com:USERNAME/USERNAME.github.io.git
	git push origin master

In my case, the `USERNAME` is `linwumeng`.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;3.   Install Jekyll<br/>
&nbsp;&nbsp;&nbsp;&nbsp;4.   Create this post by `rake`.<br/>

	$ rake post title="How do I create this blog?"

A new file named `2014-09-04-how-do-i-create-this-blog.md` is created in `_post`, which is a markdown file.

I meet problems to install Jekyll on my VM. I'm running a VM of CentOS 6.5 64-bit on Windows 7 via VMware Workstation. As the tutorial states, Jekyll is installed as

	$gem install jekyll

However, it fails due to there is no ruby on my VM at all. Hence I have to install ruby first. 

## Install Ruby
Like to install other software, I open software manager of CentOS to install Ruby. The latest ruby is 1.8.4 in the Repo. After installing of Ruby, I issued the command again to install jekyll.

Unfortunately, I got the message that the Ruby is too old to support Jekyll whic require 1.9.3 at least. Since the latest version of Ruby in the Repo is installed, I have to google again to find the solution online.

Minutes after, I found the [page](http://tecadmin.net/install-ruby-1-9-3-or-multiple-ruby-version-on-centos-6-3-using-rvm/) to address how to install Ruby 1.9.3 on CentOS 6.5.

	Step 1: Upgrade Packages
	     $ yum update 
	Step 2: Installing Recommended Packages
	     $ yum install gcc-c++ patch readline readline-devel zlib zlib-devel
	     $ yum install libyaml-devel libffi-devel openssl-devel make
	     $ yum install bzip2 autoconf automake libtool bison iconv-devel
	Step 3: Install RVM (Ruby Version Manager)
	     $ curl -L get.rvm.io | bash -s stable
	     The sampel out is like,

	       % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                              Dload  Upload   Total   Spent    Left  Speed
             100 20511  100 20511    0     0   1120      0  0:00:18  0:00:18 --:--:-- 19722
             Downloading https://github.com/wayneeseguin/rvm/archive/stable.tar.gz
             Creating group 'rvm'

             Installing RVM to /usr/local/rvm/
             Installation of RVM in /usr/local/rvm/ is almost complete:

             * First you need to add all users that will be using rvm to 'rvm' group,
               and logout - login again, anyone using rvm will be operating with `umask u=rwx,g=rwx,o=rx`.

             * To start using RVM you need to run `source /etc/profile.d/rvm.sh`
               in all your open shell windows, in rare cases you need to reopen all shell windows.

             # Administrator,
             #
             #   Thank you for using RVM!
             #   We sincerely hope that RVM helps to make your life easier and more enjoyable!!!
             #
             # ~Wayne, Michal & team.

             In case of problems: http://rvm.io/help and https://twitter.com/rvm_io
	Step 4: Setup RVM Environment
	     $ source /etc/profile.d/rvm.sh
	Step 5: Install Required Ruby Version
	     $ rvm install 1.9.3
	Step 6: Setup Default Ruvy Version
	     $ rvm use 1.9.3 --default


Now, Ruby is installed. I issued gem to install Jekyll again.

However, another error message again. The Ruby gem cannot pull data from https://rubygems.org. I needed to google as usual. Still, I found a tip at [stackoverflow](http://stackoverflow.com/questions/4418/how-do-i-update-ruby-gems-from-behind-a-prooxy-isa-htlm) where address the issue due to GFW. I try the `--http-proxy` switch of gem first, but no luck. Lastly, the workable solution is found as use the mirror of Ruby provided by Taobao.

	$ gem source -r https://rubygems.org
	$ gem source -a https://ruby.taobao.org

It's time to see the magic. It was no problem to install Jekyll.

## Install SSH public key on GitHub
I thought I had got through all the issues to block me publishing a blog, but I was so wrong. I got error message when I pushed the edited content to my Repo.

Yes, as the title, I had not install a SSH public key for the CentOS.

It's easy to find the page to tell how to do that for there is a [git help page](https://help.github.com/articles/generating-ssh-keys) to address it.

	Step 1: Check for SSH keys
	     $ ls -a ~/.ssh
	Step 2: There is no id_rsa.pub or id_dsa.pub, so I need to generate one.
	     $ ssh-keygen -t rsa -C "your_email@example.com"
	     $ ssh-agent -s
	     $ ssh-add ~/.ssh/id_rsa
	Step 3: Add the new SSH key to GitHub
	     Copy the content of ~/.ssh/id_rsa
	     Paste to GitHub

Here I can push my changes to the GitHub Repo.

## Install therubyracer

I cannot wait for anytime to preview the blog before publishing. Hence I run Jekyll on the VM as

	$ jekyll serve

What did I see? Another error message! Jekyll requires a javascript engine while I hadn't install. Thanks for the information in the error message that instructs me to find a javascript engine for Ruby on GitHub.

	$ gem install therubyracer

V8 is installed. Then open http://localhost:4000. I saw the post entry on the page and I can read it by clicking the link. According to the tip of Jekyll server, I add `--watch` swith to the command, so that I can see the changes immediately as I save them.

It's the time to enjoy the blogs.
