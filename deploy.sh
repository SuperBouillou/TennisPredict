#!/bin/bash
cd /app/tennisml
GIT_SSH_COMMAND='ssh -i ~/.ssh/github_deploy' git pull origin master
source venv/bin/activate
pip install -q -r requirements.txt
systemctl restart tennisml
echo 'Deploye !'
