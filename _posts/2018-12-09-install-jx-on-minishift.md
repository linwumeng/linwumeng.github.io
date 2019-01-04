---
title: "Install jx on minishift"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - minishift
  - jenkins
  - jx
typora-root-url: ..\
---

# Proxy

Start `lantern` on Ubuntu.

```
lantern -addr 192.168.0.105:37103
```
> **Never try**
> `lantern -addr 0.0.0.0:37103`
>
> `jx install` will fail due to `::37103`


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
--no-proxy *.svc \
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

[^]: 	"Don't try to use --docker-registry. It doesn't create and inject Kube Secrete to slave image to login docker."

6. disable proxy in the browser, then open jenkins-jx route to login `jenkins-admin` by OAuth

7. open jx project to find `jenkins-docker-cfg` secret. Reveal it to get the content, like

   ![jenkins-docker-cfg](/images/jenkins-docker-cfg-1545556182006.PNG)

8. copy the content to an online base64 tool to change the registry from `docker-registry.default.svc:5000` to `172.30.1.1:5000`

9. create secret from the json of `jenkins-docker-cfg` with new base64 encoded secret

 ```json
   apiVersion: v1
   data:
     config.json: >-
       eyJhdXRocyI6IHsiMTcyLjMwLjEuMTo1MDAwIjogeyJhdXRoIjogImMyVnlkbWxqWldGalkyOTFiblE2WlhsS2FHSkhZMmxQYVVwVFZYcEpNVTVwU1hOSmJYUndXa05KTmtscFNqa3VaWGxLY0dNelRXbFBhVXB5WkZkS2JHTnROV3hrUjFaNlRETk9iR051V25CWk1sWm9XVEpPZG1SWE5UQkphWGRwWVROV2FWcFlTblZhV0ZKc1kzazFjR0o1T1hwYVdFb3lZVmRPYkZsWFRtcGlNMVoxWkVNNWRWbFhNV3hqTTBKb1dUSlZhVTlwU25GbFEwbHpTVzEwTVZsdFZubGliVll3V2xoTmRXRlhPSFpqTWxaNVpHMXNhbHBYUm1wWk1qa3hZbTVSZG1NeVZtcGpiVll3VEcwMWFHSlhWV2xQYVVweFdsYzFjbUZYTlhwTVdHZDBZMjFXYm1GWVRqQmpibXQwWkVjNWNscFhOSFJsUnpGMVQxUlJhVXhEU25Ka1YwcHNZMjAxYkdSSFZucE1iV3gyVEROT2JHTnVXbkJaTWxab1dUSk9kbVJYTlRCTU0wNXNZMjVhY0ZreVZYUlpWMDVxWWpOV2RXUkROWFZaVnpGc1NXcHZhV0Z0Vm5WaE1teDFZM2t4TkV4WVNteGFNbXg2WkVoS05VbHBkMmxoTTFacFdsaEtkVnBZVW14amVUVndZbms1ZWxwWVNqSmhWMDVzV1ZkT2FtSXpWblZrUXpsNldsaEtNbUZYVG14TVYwWnFXVEk1TVdKdVVYVmtWMnhyU1dwdmFVOUhUVEZOTWxVMVQwUkpkRTFFV1hkT1F6QjRUVmRWTlV4WFJYbE5hbEYwVGxSSk1VNUVRWGROUkdjeVdsUkNiRWxwZDJsak0xWnBTV3B2YVdNemJIcGtSMVowVDI1T2JHTnVXbkJaTWxab1dUSk9kbVJYTlRCUGJYQTBUMjF3YkdKdGRIQmliazEwWlVNeGVWcFhaSEJqTTFKNVpWTktPUzVKWkV0ME1YRnNaekpTTVhSU2FqVnViRzVJZFdWUUxVRktOVkUxVGxoSGExZHNWbTl0TXpCSWEwVlJSVTE0UVRacVdGSmFhbUZtYzFVNE9FOUpWRnBUWjBkdk5Ga3RVRmhzTWtsbVJYRjJVVWhxVldFeFdUaEpjVFZUWDNOTldWUkJiMVJZZVhwWGEyVjNPRUZSYkZkb0xWRkdaMG8zZHpFMGIweG1PWEpMWmxjdGF6bFRNRzFJUm5CRGRXRTJOVXMxT0ZJM2FXcEdUV0k0T0hKemJYcHlXVVpZVERCYVZtUk9ielpDVDFKMk4wZ3dZV2x6VjFWVlJFTldTRVpXVFVOWGRreHJlWFJtTVVWVFVHVnZVWGRvTVROTlYyRkhTV3Q1VFV4ellUUlZiMnhFZVRKVmNVbDJaVEpSU0ZCbVJqWm5XSE5yTjJoalJUVjZZVlJaZGxkSmQxZHJXVFJ0VlhsNVQyVlBaaTF2Wm5WSWNtMTRMV3AxZFVKdGEwWmhkRFJPU0VKcVdqZFFPUzFCUzBWVWN6ZDFURlZEYTBWck9HWmFiblZ0YnpWcFpIZDFjSG90Y2pNdE56SndWbkZOTW5SdmJIUm1NbWM9In19fQ==
   kind: Secret
   metadata:
     name: jenkins-docker-login
     namespace: jx
   type: Opaque
 ```
10. open jenkins to config kubernetes plugin to use new secret `jenkins-docker-login` for `maven` node

    ![jenkins secret](/images/jenkins%20secret.PNG)

11. import your project by `jx import --url=<YOUR GIT URL>`

