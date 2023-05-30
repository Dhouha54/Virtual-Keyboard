from twilio.rest import Client



def envoyer_sms(text):
    # Vos identifiants Twilio
    account_sid = 'AC24daa9d589dbe3255da5ce89f6150650'
    auth_token = '2844636342912c85156131d8e69850e7'

    # Création de l'objet client Twilio
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=text,
        from_='+16206590197',
        to='+216 52 680 549'
    )
    print(message.sid)





























