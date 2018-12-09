---
layout: post
title: "Developing Spring Boot and AngularJS app on Heroku"
description: ""
category: 
tags: []
---
{% include JB/setup %}
#Background
We started a project using Spring Boot and AngularJS recently. I want to create an experimental project at home to explore new technologies. In the old days, I will create a VM running Linux to deploy the Web Server and Database and develop the app on the Windows since I have the developing environment on the my local box which is running Windows7. However, this solution obvisously slows down my box. Fortunately, more and more cloud providers comes. I'd like to deploy the app on Heroku this time.

#The App - SPBlog
The App I will used to experiment technologies is called SPBlog which is an code example in AngularJS course I learn recently. The app in the course is divided in to two parts, front and backend. The front part is created from scratch in the course and the backend is deployed in a cloud-platform for test purpose for all students.

This time I will combine the tow parts into one application first using Spring-Boot technologies for backend and AngularJS for the front. In addition to functionalities, I am going to utilize the best practices on code organization and testing too. I am intent to organize the front code as module-and-layer introduced by the course that divide the front into modules while each module is organized by layers of AngularJS components.

As the name indicates, SPBlog is a single page blog app. In the course the app is built upon MEAN stack, I am not going to approach so far. I will create the backend using Spring-Boot running in Jetty as REST API provider and AngularJS app as the REST API consumer.

#The architecture
On the server side, I will create a Spring Boot app which exposes the blog entries by REST API. Spring security is used to protect the pages via login form authentication. On the client side, I will use AngularJS to create single-page-app. However, the entire app may consist of multiple pages while each page uses single-page-app technologies.

I am not intent to deploy the two parts on the different servers as popular, Tomcat/Jetty for the backend and Node.js for the front. I may try it latter, but this time the two parts are combined as the single app that are all deployed in Tomcat.

