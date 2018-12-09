---
layout: post
title: "Convert PPT to PDF"
description: "a research on converting PPT to PDF."
category: 
tags: [Java Research]
---
{% include JB/setup %}
Recently, I meet two projects requiring generating or converting documents to PDF. It has been a long time since I did this last time. So I decided to do a small reserch on it first.

As a Java developer, the first idea is to find out if there are convenient Java libraries to generate or convert HTML to PDF. I found the (post)[http://www.javacodegeeks.com/2013/05/java-pdf-libraries.html] talking about the same issue. There are several libs listed in the post:
+ iText 5.0+ AGPL license
+ iText 4.2 MPL/LGPL licenses
+ PDF Box Apache License, Version 2.0
+ JPedal JPedal has a LGPL release to provide a full java PDF viewer under a LGPL license
+ FOP Apache License, Version 2.0
+ gnujpdf LGPL License
+ PJX GPLv2 License
+ PDFjet Strange Open Source license model
+ jPod BSD License
+ PDF Renderer Maintaining is not active

The conclusion is that the iText 4.2 is the best choice considering features and the license.

After that, I found another way to convert to PDF by command line, so that I don't need to write the code to do it. There are two tools I found:
+ (docs to pdf converter)[https://github.com/yeokm1/docs-to-pdf-converter]
+ (unoconv)[https://github.com/dagwieers/unoconv]

I prefer them if no code is required. unoconv seems more powerful, but it depends on OpenOffice. docs-to-pdf-converter is a standalone tool.
