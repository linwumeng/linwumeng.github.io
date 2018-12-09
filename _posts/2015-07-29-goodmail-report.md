---
layout: post
title: "GoodMail report"
description: ""
category: 
tags: []
---
{% include JB/setup %}
Code Name: GoodMail
Goal: The goal of project GoodMail aims to win the code competition by delivery a working software which provides desired functionality.
Requirement:
Scope: 1. a standalone server with one-click-startup script to provider PDF transforming service by monitoring particular mail inbox;2. a Web UI;
Architecture: With Libre Office binaries, GoodMail transforms the attachments and send them back to the original sender. Web UI provides basic functions to register mail inbox under monitoring and add authenorized mail address of ANZ to a white list to request the service. A scheduled background task polls mails from the inboxes and handles them in asynchronization way. Spring reactor project is integrated to process the mails in stream, so that the mails are processed parallelly against the filtering chain. For those needed to transform, Spring batch ensures they are handled properly and effectively in the transactional way. LO is lauched as server model during the transforming with considering effectiveness and stability by starting up and shuting down properly.
Limits: windows/HTTP/EWS API

