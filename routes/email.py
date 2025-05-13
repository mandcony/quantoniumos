"""
Quantonium OS - Email Routes

Handles email composition, sending, and retrieval through the secure resonance-encrypted mail system.
"""

from flask import request, jsonify
from flask_login import login_required, current_user
import json

from app import db
from models import User, Email
from attached_assets.resonance_encryption import resonance_encrypt, resonance_decrypt
from attached_assets.wave_primitives import WaveNumber

def email_routes(bp):
    @bp.route('/send', methods=['POST'])
    @login_required
    def send_email():
        data = request.json
        
        if not data or not data.get('recipient') or not data.get('subject') or not data.get('body'):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Find recipient user
        recipient = User.query.filter_by(email=data['recipient']).first()
        if not recipient:
            return jsonify({"error": "Recipient not found"}), 404
        
        try:
            # Create encryption wave number based on sender+recipient (for demo purposes)
            # In a real implementation, use secure key exchange
            encryption_wave = WaveNumber(1.23, 0.45)
            
            # Encrypt email contents
            encrypted_subject = resonance_encrypt(data['subject'], encryption_wave)
            encrypted_body = resonance_encrypt(data['body'], encryption_wave)
            
            # Create email record
            new_email = Email()
            new_email.sender_id = current_user.id
            new_email.recipient_id = recipient.id
            new_email.subject = encrypted_subject
            new_email.body = encrypted_body
            new_email.folder = 'sent'
            
            # Create copy for recipient's inbox
            recipient_copy = Email()
            recipient_copy.sender_id = current_user.id
            recipient_copy.recipient_id = recipient.id
            recipient_copy.subject = encrypted_subject
            recipient_copy.body = encrypted_body
            recipient_copy.folder = 'inbox'
            
            db.session.add(new_email)
            db.session.add(recipient_copy)
            db.session.commit()
            
            return jsonify({"message": "Email sent successfully"}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to send email: {str(e)}"}), 500

    @bp.route('/inbox', methods=['GET'])
    @login_required
    def get_inbox():
        try:
            # Get emails where current user is the recipient and folder is 'inbox'
            emails = Email.query.filter_by(recipient_id=current_user.id, folder='inbox').order_by(Email.created_at.desc()).all()
            
            # Decrypt and format emails
            encryption_wave = WaveNumber(1.23, 0.45)
            
            email_list = []
            for email in emails:
                sender = User.query.get(email.sender_id)
                if sender:
                    email_list.append({
                        "id": email.id,
                        "sender": {
                            "id": sender.id,
                            "email": sender.email,
                            "username": sender.username
                        },
                        "subject": resonance_decrypt(email.subject, encryption_wave),
                        "body": resonance_decrypt(email.body, encryption_wave),
                        "created_at": email.created_at.isoformat(),
                        "read": email.read
                    })
            
            return jsonify({"emails": email_list}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get inbox: {str(e)}"}), 500

    @bp.route('/sent', methods=['GET'])
    @login_required
    def get_sent():
        try:
            # Get emails where current user is the sender and folder is 'sent'
            emails = Email.query.filter_by(sender_id=current_user.id, folder='sent').order_by(Email.created_at.desc()).all()
            
            # Decrypt and format emails
            encryption_wave = WaveNumber(1.23, 0.45)
            
            email_list = []
            for email in emails:
                recipient = User.query.get(email.recipient_id)
                if recipient:
                    email_list.append({
                        "id": email.id,
                        "recipient": {
                            "id": recipient.id,
                            "email": recipient.email,
                            "username": recipient.username
                        },
                        "subject": resonance_decrypt(email.subject, encryption_wave),
                        "body": resonance_decrypt(email.body, encryption_wave),
                        "created_at": email.created_at.isoformat()
                    })
            
            return jsonify({"emails": email_list}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get sent emails: {str(e)}"}), 500

    @bp.route('/email/<email_id>', methods=['GET'])
    @login_required
    def get_email(email_id):
        try:
            # Find email
            email = Email.query.get(email_id)
            
            if not email:
                return jsonify({"error": "Email not found"}), 404
            
            # Verify ownership
            if email.recipient_id != current_user.id and email.sender_id != current_user.id:
                return jsonify({"error": "Unauthorized"}), 403
            
            # Decrypt email
            encryption_wave = WaveNumber(1.23, 0.45)
            
            # Mark as read if recipient is viewing
            if email.recipient_id == current_user.id and not email.read:
                email.read = True
                db.session.commit()
            
            # Format response based on whether user is sender or recipient
            if email.sender_id == current_user.id:
                recipient = User.query.get(email.recipient_id)
                if recipient:
                    email_data = {
                        "id": email.id,
                        "recipient": {
                            "id": recipient.id,
                            "email": recipient.email,
                            "username": recipient.username
                        },
                        "subject": resonance_decrypt(email.subject, encryption_wave),
                        "body": resonance_decrypt(email.body, encryption_wave),
                        "created_at": email.created_at.isoformat()
                    }
                else:
                    return jsonify({"error": "Recipient no longer exists"}), 404
            else:
                sender = User.query.get(email.sender_id)
                if sender:
                    email_data = {
                        "id": email.id,
                        "sender": {
                            "id": sender.id,
                            "email": sender.email,
                            "username": sender.username
                        },
                        "subject": resonance_decrypt(email.subject, encryption_wave),
                        "body": resonance_decrypt(email.body, encryption_wave),
                        "created_at": email.created_at.isoformat(),
                        "read": email.read
                    }
                else:
                    return jsonify({"error": "Sender no longer exists"}), 404
            
            return jsonify(email_data), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get email: {str(e)}"}), 500