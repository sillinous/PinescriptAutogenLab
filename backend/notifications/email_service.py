# backend/notifications/email_service.py

from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from backend.config import Config
import os
import json
from datetime import datetime


class EmailService:
    """Email notification service."""

    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.enabled = bool(self.smtp_user and self.smtp_password)

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        html: bool = False
    ) -> bool:
        """
        Send an email.

        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text or HTML)
            html: If True, send as HTML email

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            print(f"[WARN] Email not configured. Would send: {subject} to {to_email}")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email

            if html:
                part = MIMEText(body, 'html')
            else:
                part = MIMEText(body, 'plain')

            msg.attach(part)

            # Connect to SMTP server
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            print(f"[OK] Email sent to {to_email}: {subject}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return False

    def send_trade_alert(
        self,
        to_email: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        status: str
    ):
        """Send trade execution alert."""
        subject = f"Trade Alert: {side.upper()} {symbol}"

        body = f"""
Trade Executed

Symbol: {symbol}
Side: {side.upper()}
Quantity: {qty}
Price: ${price:.2f}
Status: {status.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
This is an automated message from PineLab.
        """.strip()

        return self.send_email(to_email, subject, body)

    def send_pnl_alert(
        self,
        to_email: str,
        total_pnl: float,
        win_rate: float,
        total_trades: int
    ):
        """Send daily P&L summary."""
        subject = f"Daily P&L Summary: ${total_pnl:.2f}"

        emoji = "🟢" if total_pnl > 0 else "🔴"

        body = f"""
Daily Trading Summary {emoji}

Total P&L: ${total_pnl:.2f}
Win Rate: {win_rate:.1f}%
Total Trades: {total_trades}

---
This is an automated message from PineLab.
        """.strip()

        return self.send_email(to_email, subject, body)

    def send_optimization_complete(
        self,
        to_email: str,
        strategy_name: str,
        best_params: dict,
        metric_value: float
    ):
        """Send notification when optimization completes."""
        subject = f"Optimization Complete: {strategy_name}"

        body = f"""
Strategy Optimization Complete

Strategy: {strategy_name}
Best Metric: {metric_value:.4f}

Best Parameters:
{json.dumps(best_params, indent=2)}

---
This is an automated message from PineLab.
        """.strip()

        return self.send_email(to_email, subject, body)

    def send_ab_test_result(
        self,
        to_email: str,
        test_name: str,
        winner: str,
        confidence: float
    ):
        """Send A/B test result notification."""
        subject = f"A/B Test Complete: {test_name}"

        body = f"""
A/B Test Results

Test: {test_name}
Winner: Variant {winner}
Confidence: {confidence:.1f}%

The winning variant is ready to be promoted to production.

---
This is an automated message from PineLab.
        """.strip()

        return self.send_email(to_email, subject, body)


# Global email service
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service


# Example usage and configuration notes
"""
To enable email notifications, add to your .env file:

SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password
FROM_EMAIL=noreply@pinelab.com

For Gmail:
1. Enable 2-factor authentication
2. Generate app-specific password at: https://myaccount.google.com/apppasswords
3. Use that password as SMTP_PASSWORD

For SendGrid, Mailgun, etc:
- Use their SMTP credentials
- See their documentation for SMTP_HOST and SMTP_PORT
"""
