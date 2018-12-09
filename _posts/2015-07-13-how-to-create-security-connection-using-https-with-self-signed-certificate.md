---
layout: post
title: "How to create security connection using HTTPS with self signed certificate?"
description: ""
category: 
tags: []
---
{% include JB/setup %}
HTTPS is most commonly used protocol to create security connection between client and server over internet. It adds a layer to HTTP to make the channel be security. Nowadays, most of modern browsers are capable to support the layer of SSL, TLSv1.1 and TLSv1.2. By the inventor's idea, no matter what kinds of security protocol are used for HTTPS, the principle is to use PKI mechanism to make sure that the context over internet is transported in safe and security way. So that no one can spy into the original context or inject malicious data. 

As PKI is engaged, handshaking and certificate are two pillars of HTTPS. The client and the server handshake to exchange public key enveloped by certificate signed by a CA. Handshaking protocol of HTTPS ensures that a security channel is created over internet which is open to all participants. With this security channel, it is extreme expensive to obtain the original context even using the most powerful super computer. However, it is also expensive for two endpoints of the channel to restore the original context. Hence the security channel just aims at exchange a small piece of data, the certificate. Once the certificate containing public key is exchanged via the channel, the channel is destroyed. Then the communication between the peers are over the common internet link with encrypted data by the private key. To enhance security, the exchanged certificate needs to verified by the signature of a CA. Since modern browsers and OS are distributed with well-known CAs, exchange fails if the certificate is not signed by one of these CA.

As a bank, we must protect customer's data during transporting. With the requirement of the policies and regulator rules, all applications deployed in the IT infrastructure can only transport data via HTTPS including the test environment. As a developer, we are impacted by this policies because we have to integrate the components in the test environment. Once your own component need to request data from others via HTTPS, you need the certificate of the peer. Due to its under developing, the peer component doesn't have a certificate that is signed by CA rather than a home-made certificate at most of time.

If you are lucky enough, you may get a certificate signed by our internal CA, so that the certificate can pass verification since the OS running in the IT infrastructure has already installed the CA root. Otherwise, you get a self-signed certificate created by the developing team for their internal test purpose. Since this kind of certificate is not signed by a trust CA, HTTPS handshake always fails.

After a quite while of googling, I found several answers can be grouped into two kinds. One is to user environment variables

 - javax.net.ssl.trustStore
 - javax.net.ssl.trustStorePassword
 - javax.net.ssl.keyStore
 - javax.net.ssl.ketStorePassword

Another way is to implement interfaces to ignore the certificate validation.

Although I tries these kinds of answers a lot, none of them can succeed in making connection to the endpoint requiring the certificate. Fortunately, I found that the test tool SOUP UI could make the connection after I imported the certificate to its key store. So I decide to read the code to reveal the secret.

It comes to the end of the story. The secret is that it is almost correct like the answer of implementing interfaces to ignore certificate validation. The missing part is that a critical method must to be overridden.
