#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""send_invitation_email 的纯函数测试，mock smtplib.SMTP，不真的发邮件。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.mailer import send_invitation_email

SMTP_ENV = {
    "SMTP_HOST": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USER": "bot@example.com",
    "SMTP_PASSWORD": "secret",
    "SMTP_FROM": "noreply@example.com",
}


class TestMailer(unittest.TestCase):
    @patch.dict("os.environ", SMTP_ENV, clear=True)
    @patch("ragify.core.mailer.smtplib.SMTP")
    def test_sends_via_starttls_with_correct_args(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        send_invitation_email(
            "invitee@example.com", "测试工作区", "邀请人", "http://localhost:3000/invitations/abc123"
        )

        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("bot@example.com", "secret")
        self.assertEqual(mock_server.sendmail.call_count, 1)
        args, _ = mock_server.sendmail.call_args
        self.assertEqual(args[0], "noreply@example.com")
        self.assertEqual(args[1], ["invitee@example.com"])
        self.assertIn("邀请人", args[2])
        self.assertIn("http://localhost:3000/invitations/abc123", args[2])

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_smtp_not_configured(self):
        with self.assertRaises(RuntimeError):
            send_invitation_email("x@example.com", "T", "I", "http://x/y")

    @patch.dict("os.environ", {**SMTP_ENV, "SMTP_PASSWORD": ""}, clear=True)
    def test_raises_when_partially_configured(self):
        with self.assertRaises(RuntimeError):
            send_invitation_email("x@example.com", "T", "I", "http://x/y")


if __name__ == "__main__":
    unittest.main()
