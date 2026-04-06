# utils.py

from twilio.rest import Client
import config
import time

client = Client(config.ACCOUNT_SID, config.AUTH_TOKEN)

# Cooldown tracker
_last_alert_time = {
    "fire": 0,
    "accident": 0
}
COOLDOWN_SECONDS = 30


def send_alert(message, to_number):
    try:
        client.messages.create(
            body=message,
            from_=config.TWILIO_NUMBER,
            to=to_number
        )
        print(f"[ALERT SENT] To: {to_number}")
        print(f"[MESSAGE]\n{message}")
    except Exception as e:
        print(f"[ERROR] Alert pampaledu: {e}")


def decision_and_alert(is_accident, is_fire, img_path=None):
    now = time.time()
    location_link = f"https://maps.google.com/?q={config.CAMERA_LAT},{config.CAMERA_LON}"

    if is_fire:
        if now - _last_alert_time["fire"] < COOLDOWN_SECONDS:
            print("[COOLDOWN] Fire alert skip chesham.")
            return

        msg = (
            f"🚨 FIRE ACCIDENT DETECTED!\n"
            f"Immediate action required!\n"
            f"📍 Location: {location_link}\n"
            f"🚒 Dispatch: Fire Engine + Ambulance + Police\n"
            f"📸 Proof: {img_path if img_path else 'N/A'}"
        )
        send_alert(msg, config.FIRE_NUMBER)
        _last_alert_time["fire"] = now

    elif is_accident:
        if now - _last_alert_time["accident"] < COOLDOWN_SECONDS:
            print("[COOLDOWN] Accident alert skip chesham.")
            return

        msg = (
            f"🚨 ACCIDENT DETECTED!\n"
            f"Immediate action required!\n"
            f"📍 Location: {location_link}\n"
            f"🚑 Dispatch: Ambulance + Police\n"
            f"📸 Proof: {img_path if img_path else 'N/A'}"
        )
        send_alert(msg, config.NORMAL_NUMBER)
        _last_alert_time["accident"] = now
