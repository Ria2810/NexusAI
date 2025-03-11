import os, smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv(override=True)

msg = EmailMessage()
msg['Subject'] = 'Test Email'
msg['From'] = os.environ.get('FROM_EMAIL')
print(msg['From'])
msg['To'] = 'choudhariria@gmail.com'  # Replace with your test receiver email
msg.set_content('This is a test email sent from a minimal script.')

smtp_server = os.environ.get('SMTP_SERVER')
print(smtp_server)
smtp_port = int(os.environ.get('SMTP_PORT', 465))
print(smtp_port)
smtp_username = os.environ.get('SMTP_USERNAME')
print(smtp_username)
smtp_password = os.environ.get('SMTP_PASSWORD')
print(smtp_password)

try:
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
    print("Test email sent successfully.")
except Exception as e:
    print("Error sending test email:", e)
