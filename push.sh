#!/bin/bash
 
git add .
git commit -m "1"

git config --global http.sslVerify "false"

 
# 推送到远程仓库
git push -u origin main