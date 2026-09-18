"""邀请邮件发送。纯函数，用标准库 smtplib + STARTTLS。配置从环境变量读，
未配置时直接抛异常——不像 JWT secret 那样有可以继续跑的临时兜底方案，因为
"假装发送成功但其实没发"会让邀请人以为对方收到了邮件，比明确报错更糟。

发送时用 message.as_bytes()（而不是 as_string()）——smtplib.SMTP.sendmail()
对 str 类型的消息体会强制 .encode('ascii')，一旦邀请人名/工作区名包含中文
等非 ASCII 字符（这是这个产品的常态，不是边缘情况）就会直接抛
UnicodeEncodeError 崩溃。MIMEText 对 utf-8 正文默认走 base64 编码，编码后的
内容本身就是纯 ASCII 字节，用 as_bytes() 传给 sendmail() 能完全绕开这个坑，
同时保持标准 MIME 编码（不像手动改 Content-Transfer-Encoding 为 8bit 那样
既不合规又会崩溃）。
"""

import os
import smtplib
from email.mime.text import MIMEText


def send_invitation_email(to_email: str, tenant_name: str, inviter_name: str, invite_url: str) -> None:
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM")
    if not all([host, port, user, password, sender]):
        raise RuntimeError("SMTP 未配置（需要 SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM）")

    subject = f"{inviter_name} 邀请你加入工作区 \"{tenant_name}\""
    body = f"{inviter_name} 邀请你加入 RAGify 工作区 \"{tenant_name}\"。\n\n点击链接加入：{invite_url}\n\n此链接 7 天内有效。"
    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to_email

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(sender, [to_email], message.as_bytes())
