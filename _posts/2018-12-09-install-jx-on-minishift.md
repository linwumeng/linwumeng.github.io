---
title: "Install jx on minishift"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - minishift
  - jenkins
  - jx
---

# Proxy

Start `lantern` on Ubuntu.

```
lantern -addr 192.168.0.105:37103
```

# install minishift

1. Download minishift from https://github.com/minishift/minishift/releases.
2. Create a soft link /usr/local/bin/minishift to the binary.

# create cluster

Create `ministart.sh` to start minishift with the same parameters every time.

```bash
#!/bin/bash
minishift start --cpus 4 --disk-size 40GB --memory 32GB \
--http-proxy http://192.168.0.105:37103 --https-proxy \
http://192.168.0.105:37103 \
--skip-registration --skip-startup-checks
```

Then, create the cluster by the sh file. There may be serval errors during the starting up time which can be ignored by restarting.

# install jx

Install jx by following the instructions from at https://jenkins-x.io/getting-started/install/

```
mkdir -p ~/.jx/bin
curl -L https://github.com/jenkins-x/jx/releases/download/v1.3.640/jx-linux-amd64.tar.gz | tar xzv -C ~/.jx/bin
export PATH=$PATH:~/.jx/bin
echo 'export PATH=$PATH:~/.jx/bin' >> ~/.bashrc
```

1. start minishift by `./ministart.sh`
2. `eval $(minishift oc-env)`
3. login as developer by `oc login -u developer -p dev`
4. grant developer permissions, `oc adm policy  --as system:admin add-cluster-role-to-user cluster-admin developer`
5. install packages to minishift by `jx install --provider=minishift`
6. disable proxy in the browser, then open jenkins-jx route to login `jenkins-admin` by OAuth

