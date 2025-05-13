from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

EMAILS_DIR = "emails"
os.makedirs(EMAILS_DIR, exist_ok=True)

@app.route("/send_email", methods=["POST"])
def send_email():
    """Handles sending an email and storing it."""
    data = request.get_json()
    sender = data.get("sender")
    recipient = data.get("recipient")
    subject = data.get("subject")
    body = data.get("body")

    if not all([sender, recipient, subject, body]):
        return jsonify({"error": "Missing required fields"}), 400

    recipient_file = os.path.join(EMAILS_DIR, f"{recipient}.json")

    # Read existing emails or initialize empty list
    if os.path.exists(recipient_file):
        with open(recipient_file, "r") as f:
            emails = json.load(f)
    else:
        emails = []

    emails.append({
        "sender": sender,
        "subject": subject,
        "body": body
    })

    # Save updated emails
    with open(recipient_file, "w") as f:
        json.dump(emails, f, indent=4)

    return jsonify({"message": "Email sent successfully"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
