---
layout: post
title: "How do I create the End of Day report?"
description: ""
category: 
tags: []
---
{% include JB/setup %}
## Background
We inherited a tool based on Excel from the predecessor, which can generate a simple AR list report consisting of two tables. The header table contains a little statistics per teams. The big table below the header is the AR list with data that is important to prioritize the items.

In the old days, we run the tool every working day before left the office to generate the report, and then sent it out to the management board. This end of day report is the complement of the morning report. However, it is boring and waste a bit of time cumulatively although it is a part of management process required by the management board.

## The way to escape

What is the way to escape from the boring mechanism work? Using the computer, of course, due to our programmer thinking!

1. Feasibility
   Technical feasibility is always the number one item on our check list. So no surprise that the first two questions are what and how to make it auto.
   * What?
     The report we need to send out every end of working day. As mentioned before, two tables with data items concerning by the management board.
   * How?
     I fall in love with Groovy these days. It's a brevity but powerful JVM language with many improvements to Java. As a DSL of Java language, Groovy is easy to learn and use by a Java programmer. I choose Groovy + Gradle as the develop stack a lot these days. It's natural for me to get the idea of create a Groovy app to generate the report and run the tool auto by Windows schedule task or a Jenkins job as a cron service.
     However, I told myself to restrain the impulse of writing code.
     
     **DRY, or DO NOT RE-INVENT A WHEEL.**
     
     Yes, I have to admit that most of tasks in our daily work are programmed by someone. I agree the idea on the Jenkins development wiki that try to extend an existing plugin rather than create new one if it can't satisfy you.

     Hence, I gave up the idea to create a new tool by Groovy + Gradle. To find out a way to reuse the Excel tool is the right way considering testing and backward compatibility.

     Now the feasibilty problem become concrete that wether it is possible to export the Excel sheet to HTML and send it out.
     1. Send out a mail
        No problem. The company uses Microsoft Exchange service, so it's easy to use EWS API to send mail.
     1. Export the Excel sheet
        Yes, VBA script can do it as I have already known tricks to export charts in Excel.

2. Design
   After the decision made before to reuse the Excel tool and export the report as HTML by VBA script, the high-level design is almost there.
   Architecture
   1. Excel tool to generate the report as it does now
   1. A simple VBA script to export the report as a HTML page
   1. A simple Groovy script to send out the HTML page
      1. Use Gradle to package the Groovy script as an executable jar
      1. All parameters for sending mail are stored in a property file which is passed as a CLI parameter

3. Open issue
   1. Is it fully auto that the report is sent out at a certain time as the morning report?
   1. Or the report is sent to a auditor and then
      1. Forward the mail to the management board manaually
      1. Approve the report by click a link in the mail

## Code
    {% highlight vb.net%}
    Set objExcel = CreateObject("Excel.Application")
Set objWB = objExcel.Workbooks.Open("report.xlsm")
    
objExcel.Application.Run "'report.xls'!exportHTML"
objExcel.Application.Quit
    {% endhighlight %}
