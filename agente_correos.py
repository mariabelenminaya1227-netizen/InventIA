"""Envío SMTP autenticado de informes PDF. No programa envíos."""
import os
import smtplib
from email.message import EmailMessage

def enviar_informe(destinatario, asunto, pdf):
    servidor = os.getenv("SMTP_HOST")
    puerto = int(os.getenv("SMTP_PORT","587"))
    usuario = os.getenv("SMTP_USER")
    clave = os.getenv("SMTP_PASSWORD")
    remitente = os.getenv("SMTP_FROM",usuario)
    if not all([servidor,usuario,clave,remitente]):
        raise ValueError("Configura SMTP_HOST, SMTP_PORT, SMTP_USER y SMTP_PASSWORD en .env.")
    mensaje = EmailMessage()
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje["Subject"] = asunto
    mensaje.set_content("Adjunto encontrarás el informe generado por InventIA.")
    mensaje.add_attachment(pdf,maintype="application",subtype="pdf",
                           filename="InventIA_informe.pdf")
    if puerto == 465:
        with smtplib.SMTP_SSL(servidor,puerto,timeout=25) as smtp:
            smtp.login(usuario,clave)
            smtp.send_message(mensaje)
    else:
        with smtplib.SMTP(servidor,puerto,timeout=25) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(usuario,clave)
            smtp.send_message(mensaje)
