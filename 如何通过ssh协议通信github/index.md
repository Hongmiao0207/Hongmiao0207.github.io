# 如何通过SSH协议通信Github


## 1 前置

本地和服务器经常通过HTTP协议请求github失败，为了解决该问题，因此转成SSH协议。

## 2 生成SSH密钥对

如果尚未生成SSH密钥对，可以使用以下命令在计算机上生成

```ssh
ssh-keygen -t ed25519 -C "your_email@example.com"
```

将`your_email@example.com`替换为在GitHub上注册的电子邮件地址。

当提示选择密钥的保存位置时，可以选择默认位置（通常在~/.ssh目录下），也可以选择其他位置。如果选择其他位置，请记得后续步骤中引用正确的文件路径。

## 3 添加SSH密钥到SSH代理

运行以下命令将SSH密钥添加到SSH代理，以便在GitHub上进行身份验证，替换`~/.ssh/id_ed25519`生成的密钥文件的路径

```shell
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 4 复制SSH公钥

使用以下命令来复制SSH公钥，以便将其添加到GitHub帐户中

```shell
cat ~/.ssh/id_ed25519.pub
```

## 5 添加SSH公钥到GitHub

- 登录到GitHub，然后依次转到 `Settings` > `SSH and GPG keys` > `New SSH key`。
- 在 `Title` 中，为密钥提供一个描述性的名称，以便在以后识别。
- 在 `Key` 中，粘贴您复制的SSH公钥。
- 最后，单击 `Add SSH key`。

## 6 验证SSH

为了验证SSH是否正常工作，可以运行以下命令：

```ssh
ssh -T git@github.com
```

如果配置成功则会显示 `Hi xxx! You've successfully authenticated, but GitHub does not provide shell access.`

## 7 更改项目的通信方式

在项目目录中打开命令行，检查目前的通信方式：

```shell
git remote -v

origin  https://github.com/xxx/xxx.git (fetch)
origin  https://github.com/xxx/xxx.git (push)
```

这表示我们目前的请求方式还是基于HTTP，使用下面命令改 url 链接：

```shell
git remote set-url origin  git@github.com:xxx/xxx.git   
```

验证：

```shell
git remote -v

origin  git@github.com:xxx/xxx.git (fetch)
origin  git@github.com:xxx/xxx.git (push)
```

此时已经修改为SSH了。

## 8 ssh -T git@github.com 超时的解决方案

git bash 中vim ~/.ssh/config

添加或修改内容如下：（重点第二行）

```vim
Host github.com
HostName ssh.github.com
User git
Port 443
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa
```

保存后，重新执行 `ssh -T git@github.com`

