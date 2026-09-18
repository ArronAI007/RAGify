"""邀请邮件发送。纯函数，用标准库 smtplib + STARTTLS。配置从环境变量读，
未配置时直接抛异常——不像 JWT secret 那样有可以继续跑的临时兜底方案，因为
"假装发送成功但其实没发"会让邀请人以为对方收到了邮件，比明确报错更糟。
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
    # MIMEText 对 utf-8 正文默认走 base64 编码，改成 8bit 保留明文，方便日志/测试直接查看内容。
    del message["Content-Transfer-Encoding"]
    message.set_payload(body)
    message["Content-Transfer-Encoding"] = "8bit"
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to_email

    with smtplib.SMTP(host, int(port)) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(sender, [to_email], message.as_string())
