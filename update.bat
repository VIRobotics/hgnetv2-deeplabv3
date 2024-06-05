@echo off

%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit

cd /d "%~dp0"
%YIKU_ENV%\python.exe -m pip install https://gitee.com/yiku-ai/hgnetv2-deeplabv3/releases/download/asset/yiku_seg-0.1.5-py3-none-any.whl
pause>nul
