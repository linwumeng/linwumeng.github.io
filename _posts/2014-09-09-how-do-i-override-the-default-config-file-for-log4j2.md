---
layout: post
title: "How do I override the default config file for log4j2?"
description: "I package the default config file log4j2.xml in jar and override it at deloy time for a singleton java app."
category: Lessons
tags: [lesson, tutorial, log4j2]
---
{% include JB/setup %}

## Preface
I got a bug report about the auto AR (the term for but in our organization) list notification mail tool this morning. It tells that the age of some ARs is -1. The tool is developed by me for I'm too lazy to send out the notification mail of AR list as morning report to the whole team and management layer. Hence I develop a small app by Groovy to generate the report and send it out every morning automatically.

For the line manager complained that the report was not align to the list the management layer focus on last week, I decided to reimplement it in a new way by grasping the AR list from the web page as management layer does. The old fashion is a typical Java developer method to develop the monolic-java-language app by Java API from the vendor. Although I've already considered the changes of query critera, I found that I may get different list by the API. Since I cannot afford time to investigate the reason, I decide to grasp the list from web pages by cURL in a short time for it definitely gets the same list as human does.
<!--more-->

I did learn a lot from rewriting the Groovy app leveraging cURL to grasp json, however, I want to talk about the little trouble about config file of log4j2 I meet when deploying the app.

## Choose Log4j2

I've used logback and heard the new log framework Apache Log4j2 before. I also glanced at some disputes on if Log4j2 is worth to be there. However, I have not had chance to try it.

Today, the chance comes.

As mentioned before, the bug tells the age of some ARs is -1 which should be impossible. Of course, a AR had been created before the AR list was generated. I know there must be something wrong with the time computing. As most developer do as there is no log framework, I started to debug.

Yes, blame me that I could had introduced any of Java log framework at the first time as develop the app. I didn't do that for the app is too small in Groovy script as I thought. Of course, another lesson comes, there is no small project!

Anyway, I didn't employ a log framework and lay to myself that I would add one when I refactor the app.

Murphy's Law shows again. Although I found a bug about extract AR creation time from the JSON, another one is difficult to debug for it works well on my local machine, but always get 1-offset on the remote VM which is using EST time. This is a perfect case that logging is better than debugging. I cannot move the developing environment on my local machine to the remote VM and remote debugging is painful too. I decided to use a log framework in a minute.

The question is which log framework is going to be used. Without too much thinking, I want to use SLF4j with logback or SLF4j with Log4j2. To choose one, I googled for the article to compare and contrast the two. Lastly, I chose Log4j2 for a buzz work I see these days, LMAX.

## Get Started with Log4j2

Thanks for the good documents of Apache standard for Apache Log4j2. I quickly added the four jars required to the app and make it running.

By following the document, I created the default config file `log4j2.xml` and packaged it into the jar of the app too. Everthing was ok before I deploy the final app to the remote VM.

## Override the default config file

I want to override the default config file which is packaged into the jar when I deploy the app. I cann't image the way that I have to unzip the jar and repackage it after editing the config file, also I don't want to make a new jar with the update config file for may be I want to change the log level in the future. I see the document of Log4j2 to address the issue that change the configuration dynamically, but I want to use a simpler and more static method.

The document tells that there are six steps for Log4j2 to load a config file,

	load environment var log4j.configurationFile
	load log4j-test.jsn on classpath
	load log4j-test.xml on classpath
	load log4j.jsn on classpath
	load log4j.xml on classpath
	use the default config object

Apparently, the first one is the extension point I could take.

First, I tried to set the environment variable in the Batch file

	SET log4j.configurationFile=%~dp0:0,-1%\log4j2.xml

This does not work.

Then, I tried to add the system property to the JVM by `-D` as

	java PARAMS -Dlog4j.configurationFile=%~dp0:0,-1%\log4j2.xml

This does not work too.

I'm confused. So I googled and find the answer,

	java -Dlog4j.configurationFile=%~dp0:0,-1%\log4j2.xml PARAMS

Yes, the mistake is just that `-D` is JVM option than the app parameter. Change the position to fix it.

Lastly, the app reported another error message about the URL format of `%~dp0:0,-1%\log4j2.xml`. I figured out the fix immediately as the error message showed up by add`file://` as the protocol part to form a URL as `file://%~dp0:0,-1%\log4j2.xml`.

Ok, that is all today.
